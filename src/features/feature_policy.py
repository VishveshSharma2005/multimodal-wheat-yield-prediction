"""Single source of truth for which columns may be used as model inputs.

Every trainer (stage-wise regression, quintile classification, LSTM, ablation)
imports from here so the exclusion rules cannot drift apart. Before 2026-09-05
each trainer carried its own private copy of the list and they had already
diverged.

Two categories of exclusion are enforced:

1. TARGET-DERIVED columns. Anything computed from, or whose existence depends
   on, the prediction target. `actual_available` is the important case: it is 1
   for every training and validation row and 0 for every 2024-25 forecast row,
   because the 2024-25 APY target has not been published. It therefore encodes
   "do we know the answer for this row", which is not a legitimate predictor and
   creates a train/serve distribution shift at forecast time.

2. EXACT-DUPLICATE columns. `build_stage_table.py` writes several aggregates
   under more than one name. Keeping all of them inflates the apparent feature
   count and distorts feature-importance rankings without adding information.
   Only exact duplicates verified elementwise on `stage_features.csv` are listed
   here - merely correlated features are left alone.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Identifiers, split keys and free text - never model inputs
# ---------------------------------------------------------------------------
IDENTIFIER_COLUMNS = {
    "case_id",
    "split",
    "season_year_start",
    "season_year_end",
    "season_year",
    "stage_start_date",
    "stage_end_date",
    "source_file",
    "source_sheet",
    "notes",
}

# ---------------------------------------------------------------------------
# 2. Target-derived columns - excluding these is a research-integrity guarantee
# ---------------------------------------------------------------------------
TARGET_DERIVED_COLUMNS = {
    "yield_kg_ha",
    "yield_quintile",
    "actual_yield_kg_ha",
    "predicted_yield_kg_ha",
    "lstm_predicted_yield_kg_ha",
    "absolute_error",
    "percentage_error",
    # Availability / evaluation bookkeeping. These describe whether a target
    # exists, not the agronomy of the season.
    "actual_available",
    "evaluation_status",
}

# Substring guards, applied to the lower-cased column name.
TARGET_DERIVED_SUBSTRINGS = (
    "actual_yield",
    "predicted_yield",
    "actual_available",
    "evaluation_status",
    "yield_quintile",
)

# Free-text / provenance guards.
#
# NOTE: a bare "text" token used to live in this list. It matched "texture" and
# silently dropped `image_texture_proxy_stage_avg` - a legitimate numeric image
# feature - from every model. The tokens below are matched on word boundaries
# (see `_matches_token`) so substrings of longer words no longer trigger.
FREE_TEXT_SUBSTRINGS = (
    "source",
    "note",
    "notes",
    "comment",
    "remark",
    "description",
    "free_text",
    "text",
    "sheet",
    "file",
)


def _matches_token(col_lower: str, tokens: tuple[str, ...]) -> bool:
    """Word-boundary match against `_`-separated name parts.

    `image_texture_proxy_stage_avg` -> parts {image, texture, proxy, stage, avg},
    which does not contain "text", so the column is kept.
    `source_file` -> parts {source, file}, which does match.
    """
    parts = set(col_lower.split("_"))
    return any(token in parts for token in tokens)

# ---------------------------------------------------------------------------
# 3. Exact duplicates. Key = the column kept, value = the aliases dropped.
#    Verified elementwise (including NaN pattern) on data/processed/stage_features.csv.
# ---------------------------------------------------------------------------
DUPLICATE_ALIASES = {
    "stage_start_das": ["days_after_sowing"],
    "rain_sum": ["cumulative_rain_mm", "cumulative_rainfall"],
    "tmin_mean": ["mean_tmin_c"],
    "tmax_mean": ["mean_tmax_c"],
    "gdd_cum": ["gdd_cumulative", "cumulative_gdd"],
    "ndvi_slope": ["ndvi_growth_rate"],
}

REDUNDANT_ALIAS_COLUMNS = {alias for aliases in DUPLICATE_ALIASES.values() for alias in aliases}

# Columns that legitimately index the growth stage. They are NOT target-derived
# and must stay available to the models.
PROTECTED_TEMPORAL_COLUMNS = {
    "stage",
    "stage_name",
    "stage_index",
    "stage_order",
    "stage_start_das",
    "stage_end_das",
    "crop_duration_days",
}


def is_target_derived(col: str) -> bool:
    """True if the column is computed from, or gated by, the prediction target."""
    col_lower = str(col).strip().lower()
    if col_lower in TARGET_DERIVED_COLUMNS:
        return True
    return any(token in col_lower for token in TARGET_DERIVED_SUBSTRINGS)


def is_redundant_alias(col: str) -> bool:
    """True if the column is an exact duplicate of another retained column."""
    return str(col).strip().lower() in REDUNDANT_ALIAS_COLUMNS


def is_excluded(col: str) -> bool:
    """True if `col` must not be used as a model input."""
    col_lower = str(col).strip().lower()

    if col_lower in PROTECTED_TEMPORAL_COLUMNS:
        return False

    if col_lower in IDENTIFIER_COLUMNS:
        return True
    if is_target_derived(col_lower):
        return True
    if is_redundant_alias(col_lower):
        return True
    return _matches_token(col_lower, FREE_TEXT_SUBSTRINGS)


def exclusion_reason(col: str) -> str:
    """Human-readable reason, for the feature-audit report."""
    col_lower = str(col).strip().lower()
    if col_lower in PROTECTED_TEMPORAL_COLUMNS:
        return "retained (protected temporal index)"
    if col_lower in IDENTIFIER_COLUMNS:
        return "identifier / split key"
    if is_target_derived(col_lower):
        return "target-derived (leakage / train-serve shift)"
    if is_redundant_alias(col_lower):
        keeper = next(k for k, v in DUPLICATE_ALIASES.items() if col_lower in v)
        return f"exact duplicate of '{keeper}'"
    if _matches_token(col_lower, FREE_TEXT_SUBSTRINGS):
        return "free-text / provenance"
    return "retained"


def verify_duplicates(df: pd.DataFrame) -> list[str]:
    """Re-check the DUPLICATE_ALIASES table against a dataframe.

    Returns a list of problems; empty means every declared alias really is an
    elementwise duplicate of its keeper. Used by scripts/smoke_test.py so the
    table cannot silently go stale if the feature builder changes.
    """
    problems = []
    for keeper, aliases in DUPLICATE_ALIASES.items():
        if keeper not in df.columns:
            continue
        for alias in aliases:
            if alias not in df.columns:
                continue
            if not df[keeper].equals(df[alias]):
                problems.append(f"'{alias}' is no longer an exact duplicate of '{keeper}'")
    return problems
