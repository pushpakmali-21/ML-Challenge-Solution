import pandas as pd
from .features import build_feature_df_batch, build_feature_row, FEATURE_COLUMNS


def candidates_to_feature_df(s1_prepped: pd.DataFrame, pool: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    """
    candidates: {s1_entity_id: [(cand_entity_id, cosine_sim), ...]}
    Returns a flat DataFrame: one row per (s1_id, cand_id) pair with feature columns.

    Uses vectorised batch computation — orders of magnitude faster than the
    original pair-by-pair Python loop.
    """
    return build_feature_df_batch(s1_prepped, pool, candidates)
