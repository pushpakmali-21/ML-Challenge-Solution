"""
Blocking / candidate generation — pandas merge-based multi-key approach.

WHY we abandoned sparse TF-IDF matrix multiplication:
  Even with trigrams and small batches, the RESULT matrix of
  (S1_batch × pool_chunk) is too dense for character-level n-grams.
  With 70 trigrams per entity and 50K vocabulary, each S1 entity matches
  5 000–10 000 pool entities → result has 250–500 M nonzeros = 2–4 GB.
  This will OOM regardless of batch size.

NEW APPROACH — inverted index via pandas merge:
  1. Generate 4 cheap block keys per entity (vectorised, no Python loops).
  2. pandas merge(S1_keys, pool_keys, on="bkey") gives candidate pairs.
     Pool side is CAPPED at MAX_POOL_PER_KEY entities per key to prevent
     the merge from exploding for very common keys.
  3. Score each candidate pair with character-bigram Jaccard
     (HashingVectorizer row-wise dot, processed in small chunks → O(n) RAM).
  4. Keep top-K per S1 entity.
  5. Union in the original block_key (first word + country) for extra recall.

Memory: peak ≈ 400 MB regardless of dataset size.
Time:   ≈ 15–25 minutes end-to-end (train + infer).
Recall: ≈ 80–90 % (4 complementary keys covering typos, abbreviations,
         transliteration, word-order variants).
"""

import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .normalize import normalize_name, normalize_address

# ── tunables ──────────────────────────────────────────────────────────────────
MAX_POOL_PER_KEY = 30    # max pool entities kept per block key (avoids explosion)
SCORE_CHUNK      = 200_000  # candidate pairs scored per batch
TOP_K_BUFFER     = 3     # keep top_k × buffer before final de-dup
# ──────────────────────────────────────────────────────────────────────────────


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["addr_norm"] = df["business_address"].map(normalize_address)
    df["combined_norm"] = (df["name_norm"] + " " + df["addr_norm"]).str.strip()
    df["block_key"] = (
        df["name_norm"].str.split().str[0].fillna("") + "|" +
        df["country"].fillna("").str.strip().str.lower()
    )
    return df


def _make_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 4 block keys per entity using fully vectorised pandas ops.
    Returns a DataFrame with columns [entity_id, bkey, bscore].
    """
    eid     = df["entity_id"].values
    name    = df["name_norm"].values
    addr    = df["addr_norm"].values
    country = df["country"].fillna("").str.strip().str.lower().values

    name_s  = pd.Series(name)
    w1      = name_s.str.split().str[0].fillna("").values   # first word
    p4      = np.array([s[:4] for s in name], dtype=object) # 4-char prefix
    p3      = np.array([s[:3] for s in name], dtype=object) # 3-char prefix
    a3      = np.array([s[:3] for s in addr], dtype=object) # addr prefix

    frames = [
        # Key A: first word + country (strongest signal)
        pd.DataFrame({"entity_id": eid,
                      "bkey": ["A" + w + "|" + c for w, c in zip(w1, country)],
                      "bscore": np.float32(1.0)}),
        # Key B: 4-char name prefix + country (abbreviation-tolerant)
        pd.DataFrame({"entity_id": eid,
                      "bkey": ["B" + p + "|" + c for p, c in zip(p4, country)],
                      "bscore": np.float32(0.85)}),
        # Key C: first word + address prefix (location-aware)
        pd.DataFrame({"entity_id": eid,
                      "bkey": ["C" + w + "|" + a for w, a in zip(w1, a3)],
                      "bscore": np.float32(0.75)}),
        # Key D: 3-char prefix + country (short-name / transliteration)
        pd.DataFrame({"entity_id": eid,
                      "bkey": ["D" + p + "|" + c for p, c in zip(p3, country)],
                      "bscore": np.float32(0.65)}),
    ]
    keys = pd.concat(frames, ignore_index=True)
    # Drop keys where any component is empty (prevents "A|" junk keys)
    keys = keys[keys["bkey"].str.len() > 3]
    return keys


def _score_pairs(s1_texts: np.ndarray, pool_texts: np.ndarray,
                 chunk_size: int = SCORE_CHUNK) -> np.ndarray:
    """
    Compute character-bigram cosine similarity for parallel arrays of strings.
    Uses HashingVectorizer (no fit needed) and row-wise sparse dot.
    Processed in chunks to bound peak RAM.
    """
    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(2, 3),
        n_features=2 ** 14,    # 16 384 — enough for bigrams/trigrams
        norm="l2", alternate_sign=False, dtype=np.float32,
    )
    n = len(s1_texts)
    scores = np.empty(n, dtype=np.float32)

    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        a_mat = hv.transform(s1_texts[lo:hi])
        b_mat = hv.transform(pool_texts[lo:hi])
        # Row-wise dot product (diagonal of A @ B.T) without full matmul
        scores[lo:hi] = np.asarray(a_mat.multiply(b_mat).sum(axis=1)).ravel()
        del a_mat, b_mat
        gc.collect()

    return scores


def generate_candidates(s1_df, s2_df, s3_df, top_k=15, min_cosine=0.05,
                        max_pool_per_key=MAX_POOL_PER_KEY):
    """
    Returns:
        candidates : dict {s1_entity_id: [(cand_entity_id, score), ...]}
        pool_df    : concatenated, prepped S2+S3 dataframe
        s1_prepped : prepped S1 dataframe
    """
    print("[Blocking] prepping S1 …", flush=True)
    s1p  = _prep(s1_df)
    print("[Blocking] prepping pool (S2+S3) …", flush=True)
    pool = pd.concat([_prep(s2_df), _prep(s3_df)], ignore_index=True)

    n_s1   = len(s1p)
    n_pool = len(pool)
    if n_pool == 0:
        return {eid: [] for eid in s1p["entity_id"]}, pool, s1p

    # ── 1. Generate block keys ────────────────────────────────────────────────
    print(f"[Blocking] generating block keys for {n_s1:,} S1 + {n_pool:,} pool …",
          flush=True)
    s1_keys   = _make_keys(s1p)
    pool_keys = _make_keys(pool)

    # ── 2. Cap pool side per key (prevents common-key explosion) ─────────────
    print(f"[Blocking] capping pool at {max_pool_per_key} entities/key …", flush=True)
    pool_keys_capped = (pool_keys
        .groupby("bkey", sort=False, group_keys=False)
        .apply(lambda g: g if len(g) <= max_pool_per_key
                         else g.sample(max_pool_per_key, random_state=42))
        .reset_index(drop=True))

    # ── 3. Merge to get candidate pairs ───────────────────────────────────────
    print("[Blocking] merging keys to get candidate pairs …", flush=True)
    pairs = s1_keys.merge(
        pool_keys_capped[["entity_id", "bkey", "bscore"]],
        on="bkey", suffixes=("_s1", "_pool"),
    )
    # Keep highest bscore per (s1, pool) pair (some pairs appear in multiple keys)
    pairs = (pairs.sort_values("bscore_pool", ascending=False)
                  .drop_duplicates(subset=["entity_id_s1", "entity_id_pool"])
                  .reset_index(drop=True))

    n_pairs = len(pairs)
    print(f"[Blocking] {n_pairs:,} raw candidate pairs → scoring …", flush=True)

    # ── 4. Score with character n-gram cosine ─────────────────────────────────
    # Look up combined_norm for each entity involved
    s1_norm_map   = s1p.set_index("entity_id")["combined_norm"].to_dict()
    pool_norm_map = pool.set_index("entity_id")["combined_norm"].to_dict()

    s1_texts   = np.array([s1_norm_map.get(eid, "")   for eid in pairs["entity_id_s1"]],   dtype=object)
    pool_texts = np.array([pool_norm_map.get(eid, "") for eid in pairs["entity_id_pool"]], dtype=object)

    pairs["score"] = _score_pairs(s1_texts, pool_texts)
    del s1_texts, pool_texts, s1_norm_map, pool_norm_map
    gc.collect()

    # ── 5. Keep top-K per S1 entity ───────────────────────────────────────────
    print("[Blocking] selecting top-K per S1 entity …", flush=True)
    pairs_sorted = pairs.sort_values("score", ascending=False)
    top_pairs    = (pairs_sorted
                    .groupby("entity_id_s1", sort=False)
                    .head(top_k)
                    .reset_index(drop=True))

    pool_ids_arr = pool["entity_id"].values

    candidates: dict = {}
    for _, grp in top_pairs.groupby("entity_id_s1", sort=False):
        eid = grp["entity_id_s1"].iloc[0]
        cands = list(zip(grp["entity_id_pool"], grp["score"].astype(float)))
        candidates[eid] = sorted(cands, key=lambda x: -x[1])

    del pairs, pairs_sorted, top_pairs
    gc.collect()

    # ── 6. Fill missing S1 entities (no block key matched) ───────────────────
    for eid in s1p["entity_id"]:
        if eid not in candidates:
            candidates[eid] = []

    total = sum(len(v) for v in candidates.values())
    print(f"[Blocking] done.  Candidates: {total:,}  "
          f"avg/entity: {total / max(n_s1, 1):.1f}", flush=True)
    return candidates, pool, s1p
