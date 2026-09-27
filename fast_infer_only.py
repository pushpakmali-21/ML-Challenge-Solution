"""
fast_infer_only.py — Blazing-fast inference using inverted-index lookup.

The pandas merge bottleneck is eliminated entirely:
  - Build {bkey: [pool_entity_ids]} dict once (O(pool))
  - For each S1 entity, look up its 4 keys → union of pool candidates (O(S1 × keys × cap))
  - No merge, no DataFrame join, no sorting of 50M rows

Uses the already-trained model from models_fast2/
"""
import os, re, json, pickle, time, unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict

TEST_DIR   = "dataset/test"
MODELS_DIR = "models_fast2"
OUT_DIR    = "output"
MAX_CANDS_PER_KEY = 10   # cap per key in inverted index
TOP_K = 10

# ── normalization (same as training) ──────────────────────────────────────────
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
    r"\bblvd\.?\b":"boulevard", r"\bdist\.?\b":"district", r"\bno\.?\b":"number",
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

def prep(df):
    df = df.copy()
    df["nn"] = df["business_name"].map(norm_name)
    df["na"] = df["business_address"].map(norm_addr)
    df["nc"] = df["country"].fillna("").str.strip().str.lower()
    return df

def get_keys(nn, na, nc):
    """Return list of block keys for a single entity."""
    w1 = nn.split()[0] if nn.split() else ""
    p4 = nn[:4]
    p3 = nn[:3]
    a3 = na[:3]
    keys = [
        "A" + w1 + "|" + nc,
        "B" + p4 + "|" + nc,
        "C" + w1 + "|" + a3,
        "D" + p3 + "|" + nc,
    ]
    return [k for k in keys if len(k) > 3]

# ── features ───────────────────────────────────────────────────────────────────
def _bj(a, b, n=50):
    a,b = str(a)[:n], str(b)[:n]
    if len(a)<2 and len(b)<2: return 1.0
    bga = set(a[i:i+2] for i in range(len(a)-1))
    bgb = set(b[i:i+2] for i in range(len(b)-1))
    u = bga|bgb
    return len(bga&bgb)/len(u) if u else 1.0

def _tj(a, b):
    sa,sb = set(str(a).split()), set(str(b).split())
    u = sa|sb
    return len(sa&sb)/len(u) if u else 1.0

def build_features_batch(pairs_df, s1_lookup, pool_lookup, feat_cols):
    """Vectorized feature computation for a batch of (s1_id, cand_id, score) rows."""
    s1a  = pairs_df["s1_id"].values
    ca   = pairs_df["cand_id"].values
    sims = pairs_df["score"].values.astype(np.float32)

    # Bulk lookup dicts
    sn = np.array([s1_lookup[e]["nn"] for e in s1a])
    sa = np.array([s1_lookup[e]["na"] for e in s1a])
    sc = np.array([s1_lookup[e]["nc"] for e in s1a])
    cn = np.array([pool_lookup[e]["nn"] for e in ca])
    cad= np.array([pool_lookup[e]["na"] for e in ca])
    cc = np.array([pool_lookup[e]["nc"] for e in ca])

    n = len(s1a)
    nbj=np.empty(n,np.float32); ntj=np.empty(n,np.float32)
    abj=np.empty(n,np.float32); atj=np.empty(n,np.float32)
    ntsr=np.empty(n,np.float32)
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

# ── main inference ─────────────────────────────────────────────────────────────
def run_infer():
    print("\n=== INFERENCE (inverted-index) ===", flush=True)
    t0 = time.time()

    s1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", flush=True)

    with open(f"{MODELS_DIR}/config.json") as f: cfg = json.load(f)
    with open(f"{MODELS_DIR}/model.pkl","rb") as f: clf = pickle.load(f)
    thr = cfg["threshold"]; feat_cols = cfg["feature_cols"]

    # Prep
    print("[Prep] Normalising …", flush=True)
    s1p   = prep(s1)
    pool  = pd.concat([prep(s2), prep(s3)], ignore_index=True)

    # Build lookup dicts (faster than .loc repeated calls)
    s1_lookup   = {row.entity_id: {"nn":row.nn,"na":row.na,"nc":row.nc}
                   for row in s1p.itertuples()}
    pool_lookup = {row.entity_id: {"nn":row.nn,"na":row.na,"nc":row.nc}
                   for row in pool.itertuples()}

    # Build inverted index: {bkey → list of up to MAX_CANDS_PER_KEY pool entity_ids}
    print("[Index] Building inverted index …", flush=True)
    inv_index = defaultdict(list)
    for row in pool.itertuples():
        keys = get_keys(row.nn, row.na, row.nc)
        for k in keys:
            lst = inv_index[k]
            if len(lst) < MAX_CANDS_PER_KEY:
                lst.append(row.entity_id)
    print(f"[Index] {len(inv_index):,} unique keys built in {time.time()-t0:.1f}s", flush=True)

    # Lookup candidates for each S1 entity
    print("[Cands] Generating candidates …", flush=True)
    candidates = {}
    for row in s1p.itertuples():
        keys = get_keys(row.nn, row.na, row.nc)
        seen = {}
        for k in keys:
            for cid in inv_index.get(k, []):
                seen[cid] = seen.get(cid, 0) + 1
        # Sort by hit count descending, keep top-K
        top = sorted(seen.items(), key=lambda x: -x[1])[:TOP_K]
        candidates[row.entity_id] = top
    print(f"[Cands] Done in {time.time()-t0:.1f}s", flush=True)

    # Build flat pairs DataFrame for feature computation
    print("[Features] Building features …", flush=True)
    s1_ids, cand_ids, scores = [], [], []
    for s1id, clist in candidates.items():
        for cid, sc in clist:
            s1_ids.append(s1id); cand_ids.append(cid); scores.append(sc)

    pairs_df = pd.DataFrame({"s1_id":s1_ids,"cand_id":cand_ids,"score":scores})
    feat_df  = build_features_batch(pairs_df, s1_lookup, pool_lookup, feat_cols)
    print(f"[Features] {len(feat_df):,} pairs. Scoring …", flush=True)

    if len(feat_df):
        feat_df["score"] = clf.predict_proba(feat_df[feat_cols])[:,1]
    else:
        feat_df["score"] = []

    os.makedirs(OUT_DIR, exist_ok=True)

    # candidate_pairs.tsv
    cand_rows = []
    for eid in s1["entity_id"]:
        ids = [c for c,_ in candidates.get(eid,[])]
        seen,dedup = set(),[]
        for i in ids:
            if i not in seen: seen.add(i); dedup.append(i)
        cand_rows.append((eid, ",".join(dedup)))
    pd.DataFrame(cand_rows, columns=["source1_entity_id","candidate_entity_ids"])\
      .to_csv(f"{OUT_DIR}/candidate_pairs.tsv", sep="\t", index=False)
    print("[Output] candidate_pairs.tsv written.", flush=True)

    # matching_results.tsv
    above = feat_df[feat_df["score"] >= thr]
    match_map = {}
    for _,row in above.iterrows():
        match_map.setdefault(row["source1_entity_id"],[]).append(row["candidate_entity_id"])

    cand_sets = {eid:{c for c,_ in candidates.get(eid,[])} for eid in s1["entity_id"]}
    match_rows = []
    for eid in s1["entity_id"]:
        ids = match_map.get(eid,[])
        seen,dedup = set(),[]
        for i in ids:
            if i not in seen: seen.add(i); dedup.append(i)
        dedup = [i for i in dedup if i in cand_sets.get(eid,set())]
        match_rows.append((eid, ",".join(dedup)))
    pd.DataFrame(match_rows, columns=["source1_entity_id","matched_entity_ids"])\
      .to_csv(f"{OUT_DIR}/matching_results.tsv", sep="\t", index=False)
    print("[Output] matching_results.tsv written.", flush=True)

    print(f"\n✅ Inference done in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    run_infer()
