"""
Run inference on the test set and write output/matching_results.tsv and
output/candidate_pairs.tsv in the exact challenge format.

Run:
    python3 -m src.infer --test-dir dataset/test --models-dir models --out-dir output
"""
import argparse
import pickle
import json
import pandas as pd

from .blocking import generate_candidates
from .pair_features import candidates_to_feature_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    s1 = pd.read_csv(f"{args.test_dir}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{args.test_dir}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{args.test_dir}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded test: S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    with open(f"{args.models_dir}/config.json") as f:
        config = json.load(f)
    with open(f"{args.models_dir}/model.pkl", "rb") as f:
        clf = pickle.load(f)
    threshold = config["threshold"]
    feature_columns = config["feature_columns"]
    top_k = config["top_k"]

    candidates, pool, s1_prepped = generate_candidates(s1, s2, s3, top_k=top_k)
    feat_df = candidates_to_feature_df(s1_prepped, pool, candidates)

    if len(feat_df):
        feat_df["score"] = clf.predict_proba(feat_df[feature_columns])[:, 1]
    else:
        feat_df["score"] = []

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    # --- candidate_pairs.tsv: the full candidate set fed to the model -----
    cand_rows = []
    for eid in s1["entity_id"]:
        ids = [c for c, _ in candidates.get(eid, [])]
        # de-dup while preserving order, safety net
        seen = set()
        ids = [i for i in ids if not (i in seen or seen.add(i))]
        cand_rows.append((eid, ",".join(ids)))
    cand_out = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])
    cand_out.to_csv(f"{args.out_dir}/candidate_pairs.tsv", sep="\t", index=False)

    # --- matching_results.tsv: candidates scoring >= threshold ------------
    match_map = {}
    if len(feat_df):
        winners = feat_df[feat_df["score"] >= threshold]
        for eid, grp in winners.groupby("source1_entity_id"):
            match_map[eid] = list(dict.fromkeys(grp["candidate_entity_id"]))  # de-dup, keep order

    match_rows = []
    for eid in s1["entity_id"]:
        ids = match_map.get(eid, [])
        match_rows.append((eid, ",".join(ids)))
    match_out = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])
    match_out.to_csv(f"{args.out_dir}/matching_results.tsv", sep="\t", index=False)

    n_matched = sum(1 for _, ids in match_rows if ids)
    print(f"Wrote {args.out_dir}/matching_results.tsv "
          f"({n_matched}/{len(s1)} entities matched, threshold={threshold:.2f})")
    print(f"Wrote {args.out_dir}/candidate_pairs.tsv")


if __name__ == "__main__":
    main()
