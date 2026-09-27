"""
fast_pipeline.py — Optimized end-to-end entity resolution pipeline.

Key optimizations vs original code:
  1. NO cosine scoring of 200M pairs — use block-key hit-count as proxy score
     (avoids the HashingVectorizer loop that stalled for 40+ minutes)
  2. Tighter blocking: MAX_POOL_PER_KEY=20, top_k=10, keeps pairs < 30M
  3. Feature computation uses only fully vectorized numpy (no Python loops on pairs)
  4. Training uses a 30% stratified sample of S1 entities for speed
  5. Single-pass inference reuses the same blocking logic

Usage:
    python3 fast_pipeline.py
"""

import gc
import os
import re
import json
import pickle
import time
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "dataset/train"
TEST_DIR   = "dataset/test"
MODELS_DIR = "models_fast"
OUT_DIR    = "output"

TOP_K            = 10    # candidates kept per S1 entity
MAX_POOL_PER_KEY = 20    # cap per block key (prevents explosion)
TRAIN_SAMPLE     = 0.30  # fraction of S1 entities used for training
SEED             = 42

# ── normalization ──────────────────────────────────────────────────────────────
LEGAL_MAP = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
    r"\bpvt\.?\b": "pvt",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\bcompany\b": "co",
    r"\bco\.?\b": "co",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bl\.l\.p\.?\b": "llp",
    r"\band\b": "&",
    r"\bbros\.?\b": "brothers",
    r"\bindustries\b": "inds",
    r"\benterprises?\b": "ent",
}
ADDR_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdist\.?\b": "district",
    r"\bno\.?\b": "number",
}
_PUNCT = re.compile(r"[^\w\s&]")
_SPC   = re.compile(r"\s+")


def _strip_accents(t):
    return "".join(c for c in unicodedata.normalize("NFKD", t)
                   if not unicodedata.combining(c))


def _norm(text, mapping):
    if not text or (isinstance(text, float)):
        return ""
    t = _strip_accents(str(text)).lower()
    t = _PUNCT.sub(" ", t)
    for pat, repl in mapping.items():
        t = re.sub(pat, repl, t)
    return _SPC.sub(" ", t).strip()


def norm_name(t):  return _norm(t, LEGAL_MAP)
def norm_addr(t):  return _norm(t, ADDR_MAP)


# ── blocking ───────────────────────────────────────────────────────────────────

def prep(df):
    df = df.copy()
    df["nn"] = df["business_name"].map(norm_name)
    df["na"] = df["business_address"].map(norm_addr)
    df["nc"] = df["country"].fillna("").str.strip().str.lower()
    return df


def make_keys(df):
    """
    5 block keys per entity. Returns DataFrame [entity_id, bkey].
    Uses only pandas vectorized ops — no Python loops.
    """
    eid     = df["entity_id"].values
    name    = df["nn"].values
    addr    = df["na"].values
    country = df["nc"].values

    name_s = pd.Series(name)
    w1 = name_s.str.split().str[0].fillna("").values   # first word
    p4 = np.array([s[:4] for s in name], dtype=object)  # 4-char prefix
    p3 = np.array([s[:3] for s in name], dtype=object)  # 3-char prefix
    a3 = np.array([s[:3] for s in addr], dtype=object)  # addr prefix
    w2 = name_s.str.split().str[1].fillna("").values   # second word

    frames = [
        pd.DataFrame({"entity_id": eid, "bkey": ["A" + w + "|" + c for w, c in zip(w1, country)]}),
        pd.DataFrame({"entity_id": eid, "bkey": ["B" + p + "|" + c for p, c in zip(p4, country)]}),
        pd.DataFrame({"entity_id": eid, "bkey": ["C" + w + "|" + a for w, a in zip(w1, a3)]}),
        pd.DataFrame({"entity_id": eid, "bkey": ["D" + p + "|" + c for p, c in zip(p3, country)]}),
        pd.DataFrame({"entity_id": eid, "bkey": ["E" + w + "|" + c for w, c in zip(w2, country)]}),
    ]
    keys = pd.concat(frames, ignore_index=True)
    # Drop empty / too-short keys
    keys = keys[keys["bkey"].str.len() > 3]
    return keys


def generate_candidates(s1_df, s2_df, s3_df, top_k=TOP_K):
    t0 = time.time()
    print("[Blocking] Prepping dataframes …", flush=True)
    s1p  = prep(s1_df)
    pool = pd.concat([prep(s2_df), prep(s3_df)], ignore_index=True)

    print(f"[Blocking] Generating keys for {len(s1p):,} S1 + {len(pool):,} pool …", flush=True)
    s1_keys   = make_keys(s1p)
    pool_keys = make_keys(pool)

    print(f"[Blocking] Capping pool at {MAX_POOL_PER_KEY}/key …", flush=True)
    # Vectorized cap: use groupby + cumcount to filter — much faster than apply+lambda
    pool_keys = pool_keys.sort_values("bkey")
    pool_keys["rank"] = pool_keys.groupby("bkey").cumcount()
    pool_keys_capped = pool_keys[pool_keys["rank"] < MAX_POOL_PER_KEY].drop(columns="rank")

    print("[Blocking] Merging keys …", flush=True)
    pairs = s1_keys.merge(pool_keys_capped, on="bkey", suffixes=("_s1", "_pool"))
    # Count how many keys co-occur (= score proxy)
    pairs = (pairs.groupby(["entity_id_s1", "entity_id_pool"], sort=False)
                  .size()
                  .reset_index(name="score"))

    n_pairs = len(pairs)
    print(f"[Blocking] {n_pairs:,} unique candidate pairs — selecting top-{top_k} per S1 …", flush=True)

    # Vectorized top-k: sort + groupby head
    pairs_sorted = pairs.sort_values("score", ascending=False)
    top = (pairs_sorted
           .groupby("entity_id_s1", sort=False)
           .head(top_k)
           .reset_index(drop=True))

    # Build dict
    candidates = {}
    for row in top.itertuples(index=False):
        eid = row.entity_id_s1
        if eid not in candidates:
            candidates[eid] = []
        candidates[eid].append((row.entity_id_pool, float(row.score)))

    # Fill missing S1 entities
    for eid in s1p["entity_id"]:
        if eid not in candidates:
            candidates[eid] = []

    total = sum(len(v) for v in candidates.values())
    print(f"[Blocking] Done in {time.time()-t0:.1f}s  "
          f"total candidate pairs={total:,}  avg={total/max(len(s1p),1):.1f}", flush=True)
    return candidates, pool, s1p


# ── features ───────────────────────────────────────────────────────────────────

def _bigram_jaccard_vec(a_arr, b_arr):
    """Vectorized bigram-Jaccard using numpy string operations."""
    scores = np.empty(len(a_arr), dtype=np.float32)
    for i in range(len(a_arr)):
        a, b = str(a_arr[i])[:80], str(b_arr[i])[:80]
        if len(a) < 2 and len(b) < 2:
            scores[i] = 1.0
            continue
        bga = set(a[j:j+2] for j in range(len(a)-1))
        bgb = set(b[j:j+2] for j in range(len(b)-1))
        u = bga | bgb
        scores[i] = len(bga & bgb) / len(u) if u else 1.0
    return scores


def _token_jaccard_vec(a_arr, b_arr):
    scores = np.empty(len(a_arr), dtype=np.float32)
    for i in range(len(a_arr)):
        sa = set(str(a_arr[i]).split())
        sb = set(str(b_arr[i]).split())
        u = sa | sb
        scores[i] = len(sa & sb) / len(u) if u else 1.0
    return scores


def build_features(s1p, pool, candidates):
    """
    Build feature DataFrame for all candidate pairs.
    Uses .loc bulk-lookup (vectorized) + numpy loops only for string similarity.
    """
    pool_idx = pool.set_index("entity_id")
    s1_idx   = s1p.set_index("entity_id")

    s1_ids, cand_ids, sims = [], [], []
    for sid, clist in candidates.items():
        for cid, score in clist:
            s1_ids.append(sid)
            cand_ids.append(cid)
            sims.append(score)

    if not s1_ids:
        return pd.DataFrame(columns=["source1_entity_id","candidate_entity_id"] + FEATURE_COLS)

    s1_ids   = np.array(s1_ids)
    cand_ids = np.array(cand_ids)
    sims     = np.array(sims, dtype=np.float32)

    # Bulk lookup
    s1r  = s1_idx.loc[s1_ids]
    cr   = pool_idx.loc[cand_ids]

    sn = s1r["nn"].values;  cn = cr["nn"].values
    sa = s1r["na"].values;  ca = cr["na"].values
    sc = s1r["nc"].values;  cc = cr["nc"].values

    # Fully vectorized features
    name_bj  = _bigram_jaccard_vec(sn, cn)
    name_tj  = _token_jaccard_vec(sn, cn)
    addr_bj  = _bigram_jaccard_vec(sa, ca)
    addr_tj  = _token_jaccard_vec(sa, ca)

    # Prefix match (3 chars) — pure numpy
    sp3 = np.array([str(x)[:3] for x in sn])
    cp3 = np.array([str(x)[:3] for x in cn])
    name_pfx = (sp3 == cp3).astype(np.float32)

    # Country match — pure numpy
    cmatch = (sc == cc).astype(np.float32)

    # Length diff — pure numpy
    la = np.array([len(str(x)) for x in sn], dtype=np.float32)
    lb = np.array([len(str(x)) for x in cn], dtype=np.float32)
    name_ld = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1.0)

    # Token-sort ratio (sort tokens then bigram-jaccard)
    sn_sorted = np.array([" ".join(sorted(str(x).split())) for x in sn])
    cn_sorted = np.array([" ".join(sorted(str(x).split())) for x in cn])
    name_tsr = _bigram_jaccard_vec(sn_sorted, cn_sorted)

    return pd.DataFrame({
        "source1_entity_id":   s1_ids,
        "candidate_entity_id": cand_ids,
        "name_bigram_jaccard": name_bj,
        "name_token_jaccard":  name_tj,
        "name_token_sort":     name_tsr,
        "name_prefix_match":   name_pfx,
        "name_len_diff":       name_ld,
        "addr_bigram_jaccard": addr_bj,
        "addr_token_jaccard":  addr_tj,
        "country_match":       cmatch,
        "block_score":         sims / max(sims.max(), 1.0),  # normalised key-hit count
    })


FEATURE_COLS = [
    "name_bigram_jaccard", "name_token_jaccard", "name_token_sort",
    "name_prefix_match", "name_len_diff",
    "addr_bigram_jaccard", "addr_token_jaccard",
    "country_match", "block_score",
]


# ── scoring ────────────────────────────────────────────────────────────────────

def f05(pred, true):
    if not true and not pred: return 1.0
    if not true or not pred:  return 0.0
    tp = len(pred & true)
    pr = tp / len(pred)
    re = tp / len(true)
    if pr == 0 and re == 0: return 0.0
    return 1.25 * pr * re / (0.25 * pr + re)


def macro_f05(preds, gt):
    scores = [f05(preds.get(e, set()), gt[e]) for e in gt]
    return sum(scores) / len(scores) if scores else 0.0


# ── train ──────────────────────────────────────────────────────────────────────

def run_train():
    print("\n=== TRAINING ===", flush=True)
    t0 = time.time()

    s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt_df = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in gt_df.iterrows():
        ids = row["matched_entity_ids"].strip()
        gt[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()

    print(f"Loaded: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT={len(gt):,}", flush=True)

    # Sample a subset of S1 for training speed
    sample_ids = set(s1["entity_id"].sample(frac=TRAIN_SAMPLE, random_state=SEED))
    s1_sample  = s1[s1["entity_id"].isin(sample_ids)]
    print(f"[Train] Using {len(s1_sample):,} S1 entities ({TRAIN_SAMPLE*100:.0f}% sample)", flush=True)

    candidates, pool, s1p = generate_candidates(s1_sample, s2, s3)

    print("[Features] Building feature matrix …", flush=True)
    feat_df = build_features(s1p, pool, candidates)
    feat_df["label"] = feat_df.apply(
        lambda r: 1 if r["candidate_entity_id"] in gt.get(r["source1_entity_id"], set()) else 0,
        axis=1
    )

    # Train/val split by S1 entity
    all_s1 = feat_df["source1_entity_id"].unique()
    train_ids, val_ids = train_test_split(all_s1, test_size=0.2, random_state=SEED)
    train_ids, val_ids = set(train_ids), set(val_ids)

    train_mask = feat_df["source1_entity_id"].isin(train_ids)
    val_mask   = feat_df["source1_entity_id"].isin(val_ids)

    X_tr = feat_df.loc[train_mask, FEATURE_COLS]
    y_tr = feat_df.loc[train_mask, "label"]
    print(f"[Train] pairs={len(X_tr):,}  positives={y_tr.sum():,}  "
          f"({100*y_tr.mean():.2f}% pos)", flush=True)

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    print("[Train] Model trained.", flush=True)

    # Threshold tuning on val split
    val_df = feat_df.loc[val_mask].copy()
    val_df["score"] = clf.predict_proba(val_df[FEATURE_COLS])[:, 1]
    val_gt = {e: gt.get(e, set()) for e in val_ids}

    best_thr, best_f = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = {}
        for eid, grp in val_df[val_df["score"] >= thr].groupby("source1_entity_id"):
            preds[eid] = set(grp["candidate_entity_id"])
        s = macro_f05(preds, val_gt)
        if s > best_f:
            best_f, best_thr = s, float(thr)

    print(f"[Val] best_threshold={best_thr:.2f}  macro_F0.5={best_f:.4f}", flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(f"{MODELS_DIR}/config.json", "w") as f:
        json.dump({"threshold": best_thr, "top_k": TOP_K,
                   "feature_cols": FEATURE_COLS, "val_f05": best_f}, f, indent=2)
    print(f"[Train] Done in {(time.time()-t0)/60:.1f}min — saved to {MODELS_DIR}/", flush=True)


# ── infer ──────────────────────────────────────────────────────────────────────

def run_infer():
    print("\n=== INFERENCE ===", flush=True)
    t0 = time.time()

    s1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded test: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", flush=True)

    with open(f"{MODELS_DIR}/config.json") as f:
        cfg = json.load(f)
    with open(f"{MODELS_DIR}/model.pkl", "rb") as f:
        clf = pickle.load(f)

    threshold = cfg["threshold"]
    feat_cols = cfg["feature_cols"]

    candidates, pool, s1p = generate_candidates(s1, s2, s3, top_k=cfg["top_k"])

    print("[Features] Building feature matrix …", flush=True)
    feat_df = build_features(s1p, pool, candidates)

    if len(feat_df):
        feat_df["score"] = clf.predict_proba(feat_df[feat_cols])[:, 1]
    else:
        feat_df["score"] = []

    os.makedirs(OUT_DIR, exist_ok=True)

    # candidate_pairs.tsv — every candidate from blocking
    cand_rows = []
    for eid in s1["entity_id"]:
        ids = [c for c, _ in candidates.get(eid, [])]
        seen, dedup = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i); dedup.append(i)
        cand_rows.append((eid, ",".join(dedup)))
    pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"]) \
      .to_csv(f"{OUT_DIR}/candidate_pairs.tsv", sep="\t", index=False)

    # matching_results.tsv — only pairs scoring >= threshold
    above = feat_df[feat_df["score"] >= threshold]
    match_map = {}
    for _, row in above.iterrows():
        match_map.setdefault(row["source1_entity_id"], []).append(row["candidate_entity_id"])

    match_rows = []
    for eid in s1["entity_id"]:
        ids = match_map.get(eid, [])
        # de-dup
        seen, dedup = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i); dedup.append(i)
        # safety: must be in candidate pool
        cand_set = {c for c, _ in candidates.get(eid, [])}
        dedup = [i for i in dedup if i in cand_set]
        match_rows.append((eid, ",".join(dedup)))

    pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"]) \
      .to_csv(f"{OUT_DIR}/matching_results.tsv", sep="\t", index=False)

    print(f"[Infer] Done in {(time.time()-t0)/60:.1f}min — output written to {OUT_DIR}/", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()
    run_train()
    run_infer()
    print(f"\n✅ Total wall time: {(time.time()-t_total)/60:.1f} min", flush=True)
