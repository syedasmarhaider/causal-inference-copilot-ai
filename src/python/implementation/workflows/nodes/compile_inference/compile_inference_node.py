from __future__ import annotations
from typing import List, Set

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
import pandas as pd



def edit_df_drop_cols_expect_required(
    df: pd.DataFrame,
    compiled_protocol: ProtocolSpec,
    *,
    keep_all_original: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    required: Set[str] = set()

    # time zero
    if getattr(compiled_protocol, "time_zero_type", None) == "COLUMN":
        required.add(compiled_protocol.time_zero)

    # exclusions
    for ex in compiled_protocol.exclusions:
        required.add(ex.column)

    # treatment column
    required.add(compiled_protocol.treatment_spec.column)

    # outcome column(s)
    ys = compiled_protocol.outcome_spec
    # duration has two columns; others have .column
    if getattr(ys, "kind", None) == "duration":
        required.add(getattr(ys, "duration_column"))
        required.add(getattr(ys, "event_column"))
    else:
        required.add(getattr(ys, "column"))

    # covariates / effect modifiers
    required.update(list(compiled_protocol.covariates))
    required.update(list(compiled_protocol.effect_modifiers))

    # normalize: strip + remove empties (shouldn’t happen due to NonEmptyStr, but defensive)
    required = {c.strip() for c in required if c.strip()}

    df_cols = [str(c) for c in df.columns]
    df_col_set = set(df_cols)

    missing = sorted([c for c in required if c not in df_col_set])
    required_sorted = sorted(required)

    if strict and missing:
        raise ValueError(f"edit_df_drop_cols_expect_required: missing required columns: {missing}")

    if keep_all_original:
        kept = df_cols
        dropped: List[str] = []
        out = df.copy()
    else:
        kept = [c for c in df_cols if c in required]
        dropped = [c for c in df_cols if c not in required]
        out = df.loc[:, kept].copy()

    summary = DropColsSummary(
        kept_cols=kept,
        dropped_cols=dropped,
        missing_required=missing,
        required_cols=required_sorted,
    )
    return out, summary
    
    
    
   