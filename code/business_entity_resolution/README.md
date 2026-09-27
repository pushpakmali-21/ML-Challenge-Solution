# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline: normalization → TF-IDF/cosine blocking → pairwise
feature engineering → gradient-boosted classifier → F_0.5-optimal
thresholding → challenge-format TSV output.

Everything runs on `pandas` / `numpy` / `scipy` / `scikit-learn` only —
all MIT/BSD licensed, no external APIs or data lookups, satisfying the
challenge's fair-play and model-license rules (well under 8B params).

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Put the real challenge data in place

Copy the actual challenge files into:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

(This repo ships with tiny synthetic placeholder files of the same shape
so you can smoke-test the code immediately — replace them with the real
data before you care about the actual score.)

## 3. Train

```bash
python3 -m src.train_model --train-dir dataset/train --models-dir models
```

This will:
- Run blocking on the full training pool and print the **recall ceiling**
  and **reduction ratio** (how good the candidate generation stage is —
  this is the upper bound on achievable recall).
- Split Source-1 entities 80/20 into train/validation (never leaks across
  the split).
- Train a `HistGradientBoostingClassifier` on pairwise similarity features.
- Sweep decision thresholds on the validation split and pick the one that
  **maximizes macro F_0.5** (the actual competition metric), not F1 or
  accuracy.
- Save `models/model.pkl` and `models/config.json` (threshold + feature list).

## 4. Run inference on the test set

```bash
python3 -m src.infer --test-dir dataset/test --models-dir models --out-dir output
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` in
the exact challenge format.

## 5. Validate before submitting

Run the challenge's own validator against your output:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

(`utils/validate_submission.py` ships with the challenge's
`student_resource/` package — copy it in here, or run from that directory
pointing `--out-dir` at this repo's `output/`.)

## Pipeline design

### Blocking (`src/blocking.py`)
Character n-gram (2–4) TF-IDF over `normalized_name + normalized_address`,
fit once on the combined Source2+Source3 pool. Cosine similarity (sparse
matrix multiply) picks the top-K candidates per Source-1 entity — this
catches typos, abbreviations, and transliteration because it works at the
character level, not the token level, so it's robust to noise standard
token-overlap blocking would miss. A cheap block-key union (first name
token + country) is added on top purely to backstop recall for
near-duplicate clusters. Country is *never* used as a hard filter, only
as one signal among several, so unseen test countries (e.g. France) never
get silently dropped from candidate generation.

### Features (`src/features.py`)
For each (Source-1, candidate) pair: normalized Levenshtein ratio, token-
sort ratio (handles word-order transpositions), Jaccard token overlap,
prefix match, length-difference ratio — each computed separately for name
and address — plus a country-match flag and the blocking-stage cosine
similarity itself as a feature.

### Model (`src/train_model.py`)
`sklearn.ensemble.HistGradientBoostingClassifier` (BSD-licensed, tiny,
well under the 8B-parameter cap) trained on labeled candidate pairs
(positive = true match from ground truth, negative = any other candidate
the blocking stage surfaced). Class-balanced to handle the heavy negative
skew inherent to candidate pairs.

### Thresholding
Because F_0.5 weights precision 2x over recall, and singletons need to be
predicted as empty lists to score 1.0, the threshold is chosen by directly
maximizing macro F_0.5 on a held-out validation split — not by a generic
0.5 cutoff or by optimizing F1/accuracy.

## Next steps to push the score further once real data is loaded
- Inspect `models/config.json`'s `blocking_recall_ceiling` — if it's not
  close to 1.0, that's your bottleneck, not the classifier. Raise `top_k`
  or loosen `min_cosine` in `generate_candidates`.
- Add a sentence-embedding similarity feature (e.g. an Apache-2.0
  `sentence-transformers` model such as `all-MiniLM-L6-v2`) for semantic
  name/address matching beyond surface string similarity, if network
  access to download model weights is allowed under the challenge's
  fair-play rules (it is not an "external lookup service" — it doesn't
  call out per-example at inference/training time — but confirm against
  the challenge's exact wording before using it).
- Add per-country address-format-aware features (e.g. PIN/ZIP code
  presence/match) since the prompt explicitly calls out country-specific
  address patterns.
- Try calibrating per-country thresholds separately if error patterns
  differ meaningfully between US/India/France.
