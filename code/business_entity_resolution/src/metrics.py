"""
F_beta (beta=0.5) macro-average scorer, matching the challenge's exact
evaluation definition: computed per Source-1 entity, then averaged.
A correctly-predicted empty list (true singleton) scores 1.0; predicting
any match for a true singleton scores 0.0.
"""


def f_beta_per_entity(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if not pred_ids and true_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f_beta(predictions: dict, ground_truth: dict, beta: float = 0.5) -> float:
    """
    predictions / ground_truth: {source1_entity_id: set(matched_ids)}
    Averages over every key present in ground_truth (as the challenge does).
    """
    scores = []
    for eid, true_ids in ground_truth.items():
        pred_ids = predictions.get(eid, set())
        scores.append(f_beta_per_entity(pred_ids, true_ids, beta))
    return sum(scores) / len(scores) if scores else 0.0
