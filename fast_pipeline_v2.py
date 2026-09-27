"""
fast_pipeline_v2.py — Aggressive speed optimization.

Key changes from v1:
  - Training uses only 5% S1 sample (still statistically representative)
  - MAX_POOL_PER_KEY reduced to 10 → merge produces ~10M pairs not 55M
  - Candidate dict built from groupby+apply (no itertuples loop)
  - Feature loop capped: bigram jaccard only on first 50 chars
  - Inference blocking runs the same tight settings
"""

import gc, os, re, json, pickle, time, unicodedata
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

TRAIN_DIR  = "dataset/train"
TEST_DIR   = "dataset/test"
MODELS_DIR = "models_fast2"
OUT_DIR    = "output"

TOP_K            = 10
MAX_POOL_PER_KEY = 10    # was 30 (original) → 20 (v1) → 10 now
TRAIN_SAMPLE     = 0.05  # 5% of 2.2M = 110K entities → fast blocking
SEED             = 42

# ── normalization ──────────────────────────────────────────────────────────────
LEGAL_MAP = {
    r"\bprivate limited\b":"pvt ltd", r"\bpvt\.?\s*ltd\.?\b":"pvt ltd",
    r"\bpvt\.?\b":"pvt", r"\blimited\b":"ltd", r"\bltd\.?\b":"ltd",
    r"\bcorporation\b":"corp", r"\bcorp\.?\b":"corp",
    r"\bincorporated\b":"inc", r"\binc\.?\b":"inc",
    r"\bcompany\b":"co", r"\bco\.?\b":"co",
    r"\bl\.l\.c\.?\b":"llc", r"\bl\.l\.p\.?\b":"llp",
    r"\band\b":"&", r"\bbros\.?\b":"brothers",
    r"\bindustries\b":"inds", r"\benterprises?\b":"ent",
}
ADDR_MAP = {
    r"\brd\.?\b":"road", r"\bst\.?\b":"street", r"\bave\.?\b":"avenue",
    r"\bblvd\.?\b":"boulevard", r"\bdist\.?\b":"district",
    r"\bno\.?\b":"number",
}
_PUNCT = re.compile(r"[^\w\s&]")
_SPC   = re.compile(r"\s+")

def _strip_accents(t):
    return "".join(c for c in unicodedata.normalize("NFKD",t) if not unicodedata.combining(c))

def _norm(text, mapping):
    if not text or isinstance(text, float): return ""
    t = _strip_accents(str(text)).lower()
    t = _PUNCT.sub(" ", t)
    for p,r in mapping.items(): t = re.sub(p,r,t)
    return _SPC.sub(" ",t).strip()

def norm_name(t): return _norm(t, LEGAL_MAP)
def norm_addr(t): return _norm(t, ADDR_MAP)

# ── blocking ───────────────────────────────────────────────────────────────────

def prep(df):
    df = df.copy()
    df["nn"] = df["business_name"].map(norm_name)
    df["na"] = df["business_address"].map(norm_addr)
    df["nc"] = df["country"].fillna("").str.strip().str.lower()
    return df

def make_keys(df):
    eid     = df["entity_id"].values
    name    = df["nn"].values
    addr    = df["na"].values
    country = df["nc"].values
    ns = pd.Series(name)
    w1 = ns.str.split().str[0].fillna("").values
    p4 = np.array([s[:4] for s in name], dtype=object)
    p3 = np.array([s[:3] for s in name], dtype=object)
    a3 = np.array([s[:3] for s in addr], dtype=object)
    frames = [
        pd.DataFrame({"entity_id":eid, "bkey":["A"+w+"|"+c for w,c in zip(w1,country)]}),
        pd.DataFrame({"entity_id":eid, "bkey":["B"+p+"|"+c for p,c in zip(p4,country)]}),
        pd.DataFrame({"entity_id":eid, "bkey":["C"+w+"|"+a for w,a in zip(w1,a3)]}),
        pd.DataFrame({"entity_id":eid, "bkey":["D"+p+"|"+c for p,c in zip(p3,country)]}),
    ]
    keys = pd.concat(frames, ignore_index=True)
    return keys[keys["bkey"].str.len() > 3]

def generate_candidates(s1_df, s2_df, s3_df, top_k=TOP_K, max_ppk=MAX_POOL_PER_KEY):
    t0 = time.time()
    print("[Blocking] Prepping …", flush=True)
    s1p  = prep(s1_df)
    pool = pd.concat([prep(s2_df), prep(s3_df)], ignore_index=True)

    print(f"[Blocking] Keys: {len(s1p):,} S1 + {len(pool):,} pool …", flush=True)
    s1_keys   = make_keys(s1p)
    pool_keys = make_keys(pool)

    # Cap pool per key using cumcount (fully vectorized)
    pool_keys = pool_keys.sort_values("bkey")
    pool_keys["rk"] = pool_keys.groupby("bkey").cumcount()
    pool_capped = pool_keys[pool_keys["rk"] < max_ppk].drop(columns="rk")

    print("[Blocking] Merging …", flush=True)
    pairs = s1_keys.merge(pool_capped, on="bkey", suffixes=("_s1","_pool"))

    print(f"[Blocking] {len(pairs):,} raw pairs → groupby score …", flush=True)
    # Score = number of keys that match
    scored = (pairs.groupby(["entity_id_s1","entity_id_pool"], sort=False)
                   .size()
                   .reset_index(name="score"))

    print(f"[Blocking] {len(scored):,} unique pairs → top-{top_k} …", flush=True)
    # Vectorized top-k via nlargest per group
    top = (scored.sort_values("score", ascending=False)
                 .groupby("entity_id_s1", sort=False)
                 .head(top_k)
                 .reset_index(drop=True))

    # Build candidates dict using groupby — no itertuples
    cand_dict = {
        eid: list(zip(grp["entity_id_pool"], grp["score"].astype(float)))
        for eid, grp in top.groupby("entity_id_s1", sort=False)
    }
    for eid in s1p["entity_id"]:
        cand_dict.setdefault(eid, [])

    total = sum(len(v) for v in cand_dict.values())
    print(f"[Blocking] Done {time.time()-t0:.1f}s  pairs={total:,}  avg={total/max(len(s1p),1):.1f}", flush=True)
    return cand_dict, pool, s1p

# ── features ───────────────────────────────────────────────────────────────────

def _bj(a, b, n=50):
    a,b = str(a)[:n], str(b)[:n]
    if len(a)<2 and len(b)<2: return 1.0
    bga = set(a[i:i+2] for i in range(len(a)-1))
    bgb = set(b[i:i+2] for i in range(len(b)-1))
    u = bga|bgb
    return len(bga&bgb)/len(u) if u else 1.0

def _tj(a,b):
    sa,sb = set(str(a).split()), set(str(b).split())
    u = sa|sb
    return len(sa&sb)/len(u) if u else 1.0

def build_features(s1p, pool, candidates):
    pool_idx = pool.set_index("entity_id")
    s1_idx   = s1p.set_index("entity_id")
    s1_ids, cand_ids, sims = [], [], []
    for sid, clist in candidates.items():
        for cid, sc in clist:
            s1_ids.append(sid); cand_ids.append(cid); sims.append(sc)
    if not s1_ids:
        return pd.DataFrame(columns=["source1_entity_id","candidate_entity_id"]+FEATURE_COLS)

    s1a  = np.array(s1_ids);  ca  = np.array(cand_ids)
    sims = np.array(sims, dtype=np.float32)

    s1r = s1_idx.loc[s1a];  cr = pool_idx.loc[ca]
    sn,cn = s1r["nn"].values, cr["nn"].values
    sa,cad = s1r["na"].values, cr["na"].values
    sc,cc  = s1r["nc"].values, cr["nc"].values

    n = len(s1a)
    nbj = np.empty(n,np.float32); ntj = np.empty(n,np.float32)
    abj = np.empty(n,np.float32); atj = np.empty(n,np.float32)
    ntsr= np.empty(n,np.float32)
    for i in range(n):
        nbj[i]  = _bj(sn[i],cn[i])
        ntj[i]  = _tj(sn[i],cn[i])
        abj[i]  = _bj(sa[i],cad[i])
        atj[i]  = _tj(sa[i],cad[i])
        ntsr[i] = _bj(" ".join(sorted(str(sn[i]).split())),
                       " ".join(sorted(str(cn[i]).split())))

    sp3 = np.array([str(x)[:3] for x in sn])
    cp3 = np.array([str(x)[:3] for x in cn])
    npfx = (sp3==cp3).astype(np.float32)
    cmatch = (sc==cc).astype(np.float32)
    la = np.array([len(str(x)) for x in sn],np.float32)
    lb = np.array([len(str(x)) for x in cn],np.float32)
    nld = np.abs(la-lb)/np.maximum(np.maximum(la,lb),1.0)

    mx = max(sims.max(),1.0)
    return pd.DataFrame({
        "source1_entity_id":s1a, "candidate_entity_id":ca,
        "name_bj":nbj, "name_tj":ntj, "name_tsr":ntsr,
        "name_pfx":npfx, "name_ld":nld,
        "addr_bj":abj, "addr_tj":atj,
        "country_match":cmatch, "block_score":sims/mx,
    })

FEATURE_COLS = ["name_bj","name_tj","name_tsr","name_pfx","name_ld",
                "addr_bj","addr_tj","country_match","block_score"]

# ── f0.5 ───────────────────────────────────────────────────────────────────────

def f05(pred,true):
    if not true and not pred: return 1.0
    if not true or  not pred: return 0.0
    tp=len(pred&true); pr=tp/len(pred); re=tp/len(true)
    if pr==0 and re==0: return 0.0
    return 1.25*pr*re/(0.25*pr+re)

def macro_f05(preds,gt):
    scores=[f05(preds.get(e,set()),gt[e]) for e in gt]
    return sum(scores)/len(scores) if scores else 0.0

# ── train ──────────────────────────────────────────────────────────────────────

def run_train():
    print("\n=== TRAINING ===",flush=True)
    t0=time.time()
    s1=pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv",sep="\t",dtype=str,keep_default_na=False)
    s2=pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv",sep="\t",dtype=str,keep_default_na=False)
    s3=pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv",sep="\t",dtype=str,keep_default_na=False)
    gt_df=pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv",sep="\t",dtype=str,keep_default_na=False)
    gt={}
    for _,row in gt_df.iterrows():
        ids=row["matched_entity_ids"].strip()
        gt[row["source1_entity_id"]]=set(ids.split(",")) if ids else set()
    print(f"Loaded: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}",flush=True)

    s1_samp=s1.sample(frac=TRAIN_SAMPLE,random_state=SEED)
    print(f"[Train] Sample: {len(s1_samp):,} S1 entities ({TRAIN_SAMPLE*100:.0f}%)",flush=True)

    cands,pool,s1p=generate_candidates(s1_samp,s2,s3)

    print("[Features] Building …",flush=True)
    feat=build_features(s1p,pool,cands)
    feat["label"]=feat.apply(
        lambda r: 1 if r["candidate_entity_id"] in gt.get(r["source1_entity_id"],set()) else 0, axis=1)

    all_s1=feat["source1_entity_id"].unique()
    tr_ids,val_ids=train_test_split(all_s1,test_size=0.2,random_state=SEED)
    tr_ids,val_ids=set(tr_ids),set(val_ids)

    tmask=feat["source1_entity_id"].isin(tr_ids)
    vmask=feat["source1_entity_id"].isin(val_ids)
    Xtr=feat.loc[tmask,FEATURE_COLS]; ytr=feat.loc[tmask,"label"]
    print(f"[Train] pairs={len(Xtr):,} pos={ytr.sum():,} ({100*ytr.mean():.2f}%)",flush=True)

    clf=HistGradientBoostingClassifier(
        max_iter=300,learning_rate=0.06,max_depth=6,
        l2_regularization=1.0,class_weight="balanced",random_state=SEED)
    clf.fit(Xtr,ytr)
    print("[Train] Model fitted.",flush=True)

    vdf=feat.loc[vmask].copy()
    vdf["score"]=clf.predict_proba(vdf[FEATURE_COLS])[:,1]
    vgt={e:gt.get(e,set()) for e in val_ids}
    best_thr,best_f=0.5,-1.0
    for thr in np.arange(0.05,0.96,0.05):
        preds={}
        for eid,grp in vdf[vdf["score"]>=thr].groupby("source1_entity_id"):
            preds[eid]=set(grp["candidate_entity_id"])
        s=macro_f05(preds,vgt)
        if s>best_f: best_f,best_thr=s,float(thr)
    print(f"[Val] thr={best_thr:.2f}  F0.5={best_f:.4f}",flush=True)

    os.makedirs(MODELS_DIR,exist_ok=True)
    with open(f"{MODELS_DIR}/model.pkl","wb") as f: pickle.dump(clf,f)
    with open(f"{MODELS_DIR}/config.json","w") as f:
        json.dump({"threshold":best_thr,"top_k":TOP_K,"max_ppk":MAX_POOL_PER_KEY,
                   "feature_cols":FEATURE_COLS,"val_f05":best_f},f,indent=2)
    print(f"[Train] Done {(time.time()-t0)/60:.1f}min",flush=True)

# ── infer ──────────────────────────────────────────────────────────────────────

def run_infer():
    print("\n=== INFERENCE ===",flush=True)
    t0=time.time()
    s1=pd.read_csv(f"{TEST_DIR}/test_source1.tsv",sep="\t",dtype=str,keep_default_na=False)
    s2=pd.read_csv(f"{TEST_DIR}/test_source2.tsv",sep="\t",dtype=str,keep_default_na=False)
    s3=pd.read_csv(f"{TEST_DIR}/test_source3.tsv",sep="\t",dtype=str,keep_default_na=False)
    print(f"Test: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}",flush=True)

    with open(f"{MODELS_DIR}/config.json") as f: cfg=json.load(f)
    with open(f"{MODELS_DIR}/model.pkl","rb") as f: clf=pickle.load(f)
    thr=cfg["threshold"]; fcols=cfg["feature_cols"]

    cands,pool,s1p=generate_candidates(s1,s2,s3,top_k=cfg["top_k"],max_ppk=cfg["max_ppk"])

    print("[Features] Building …",flush=True)
    feat=build_features(s1p,pool,cands)
    if len(feat): feat["score"]=clf.predict_proba(feat[fcols])[:,1]
    else:         feat["score"]=[]

    os.makedirs(OUT_DIR,exist_ok=True)

    # candidate_pairs.tsv
    cand_rows=[]
    for eid in s1["entity_id"]:
        ids=[c for c,_ in cands.get(eid,[])]
        seen,dedup=set(),[]
        for i in ids:
            if i not in seen: seen.add(i);dedup.append(i)
        cand_rows.append((eid,",".join(dedup)))
    pd.DataFrame(cand_rows,columns=["source1_entity_id","candidate_entity_ids"])\
      .to_csv(f"{OUT_DIR}/candidate_pairs.tsv",sep="\t",index=False)

    # matching_results.tsv
    above=feat[feat["score"]>=thr]
    match_map={}
    for _,row in above.iterrows():
        match_map.setdefault(row["source1_entity_id"],[]).append(row["candidate_entity_id"])

    cand_sets={eid:{c for c,_ in cands.get(eid,[])} for eid in s1["entity_id"]}
    match_rows=[]
    for eid in s1["entity_id"]:
        ids=match_map.get(eid,[])
        seen,dedup=set(),[]
        for i in ids:
            if i not in seen: seen.add(i);dedup.append(i)
        dedup=[i for i in dedup if i in cand_sets.get(eid,set())]
        match_rows.append((eid,",".join(dedup)))
    pd.DataFrame(match_rows,columns=["source1_entity_id","matched_entity_ids"])\
      .to_csv(f"{OUT_DIR}/matching_results.tsv",sep="\t",index=False)

    print(f"[Infer] Done {(time.time()-t0)/60:.1f}min — output in {OUT_DIR}/",flush=True)

# ── main ───────────────────────────────────────────────────────────────────────

if __name__=="__main__":
    t=time.time()
    run_train()
    run_infer()
    print(f"\n✅ Total: {(time.time()-t)/60:.1f} min",flush=True)
