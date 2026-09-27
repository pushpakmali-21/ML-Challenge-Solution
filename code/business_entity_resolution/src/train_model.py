"""
Train the matching classifier and pick a decision threshold that maximizes
macro F_0.5 on a held-out validation split of Source-1 entities.

Run:
    python3 -m src.train_model --train-dir dataset/train --models-dir models
"""
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .blocking import generate_candidates
from .pair_features import candidates_to_feature_df
from .features import FEATURE_COLUMNS
from .metrics import macro_f_beta


def load_ground_truth(path):
    gt_df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt = {}
    for _, row in gt_df.iterrows():
        ids = row["matched_entity_ids"].strip()
        gt[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1 = pd.read_csv(f"{args.train_dir}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{args.train_dir}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{args.train_dir}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = load_ground_truth(f"{args.train_dir}/train_ground_truth.tsv")

    print(f"Loaded train: S1={len(s1)} S2={len(s2)} S3={len(s3)} ground_truth={len(gt)}")

    # Split BY Source-1 entity so validation truly measures generalization.
    s1_ids = s1["entity_id"].tolist()
    train_ids, val_ids = train_test_split(s1_ids, test_size=args.val_frac, random_state=args.seed)
    train_ids, val_ids = set(train_ids), set(val_ids)

    # Blocking is run once against the full candidate pool (S2+S3), mirroring
    # exactly how it will run at test time.
    candidates, pool, s1_prepped = generate_candidates(s1, s2, s3, top_k=args.top_k)

    # --- Recall ceiling / reduction ratio report (blocking quality) ---
    total_true_links = sum(len(v) for v in gt.values())
    recovered = 0
    total_candidate_pairs = 0
    for eid, cand_list in candidates.items():
        cand_ids = {c for c, _ in cand_list}
        total_candidate_pairs += len(cand_ids)
        recovered += len(cand_ids & gt.get(eid, set()))
    recall_ceiling = recovered / total_true_links if total_true_links else 1.0
    reduction_ratio = 1 - (total_candidate_pairs / (len(s1) * (len(s2) + len(s3)))) if len(s1) else 0.0
    print(f"[Blocking] recall ceiling = {recall_ceiling:.4f}  "
          f"reduction ratio = {reduction_ratio:.6f}  "
          f"avg candidates/entity = {total_candidate_pairs/max(len(s1),1):.1f}")

    feat_df = candidates_to_feature_df(s1_prepped, pool, candidates)
    feat_df["label"] = feat_df.apply(
        lambda r: 1 if r["candidate_entity_id"] in gt.get(r["source1_entity_id"], set()) else 0, axis=1
    )

    train_mask = feat_df["source1_entity_id"].isin(train_ids)
    val_mask = feat_df["source1_entity_id"].isin(val_ids)

    X_train = feat_df.loc[train_mask, FEATURE_COLUMNS]
    y_train = feat_df.loc[train_mask, "label"]
    print(f"[Train] pairs={len(X_train)} positives={y_train.sum()} "
          f"({100*y_train.mean():.2f}% positive rate)")

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, class_weight="balanced", random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    # --- Threshold tuning on validation split for macro F_0.5 -------------
    val_df = feat_df.loc[val_mask].copy()
    if len(val_df):
        val_df["score"] = clf.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    else:
        val_df["score"] = []

    val_gt = {eid: ids for eid, ids in gt.items() if eid in val_ids}
    # ensure every val S1 entity is represented (even with zero candidates)
    for eid in val_ids:
        val_gt.setdefault(eid, gt.get(eid, set()))

    best_thr, best_f05 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        preds = {}
        for eid, grp in val_df[val_df["score"] >= thr].groupby("source1_entity_id"):
            preds[eid] = set(grp["candidate_entity_id"])
        score = macro_f_beta(preds, val_gt, beta=0.5)
        if score > best_f05:
            best_f05, best_thr = score, float(thr)

    print(f"[Validation] best threshold={best_thr:.2f}  macro F_0.5={best_f05:.4f}")

    import os
    os.makedirs(args.models_dir, exist_ok=True)
    with open(f"{args.models_dir}/model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(f"{args.models_dir}/config.json", "w") as f:
        json.dump({"threshold": best_thr, "top_k": args.top_k,
                   "feature_columns": FEATURE_COLUMNS,
                   "val_macro_f05": best_f05,
                   "blocking_recall_ceiling": recall_ceiling}, f, indent=2)
    print(f"Saved model + config to {args.models_dir}/")


if __name__ == "__main__":
    main()
