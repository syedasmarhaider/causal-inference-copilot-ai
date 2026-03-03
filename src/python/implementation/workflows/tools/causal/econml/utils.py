
from __future__ import annotations
from datetime import datetime, timezone
import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type
import numpy as np
import pandas as pd

from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.causal.encoding_plan import CatOneHotParams, DateTimeEpochParams, DropParams, EncodingPresetSpec, MapBinaryParams, MapOrdinalParams, NumLog1pParams, NumMinMaxParams, NumStandardParams, PassthroughParams, TransformPlan


class ModelSpecError(ValueError):
    pass


# a hack
_EMPTY = inspect._empty # pyright: ignore[reportPrivateUsage]

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def jsonish_default(v: Any) -> Any:
    """Return JSON-friendly default if possible; otherwise repr()."""
    if v is _EMPTY:
        return None
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)) and all(
        x is None or isinstance(x, (bool, int, float, str)) for x in v # pyright: ignore[reportUnknownVariableType]
    ):
        return list(v)  # type: ignore[return-value]
    if isinstance(v, dict):
        # keep dict only if it is JSON-ish
        ok = True
        out: Dict[str, Any] = {}
        for k, vv in v.items(): # pyright: ignore[reportUnknownVariableType]
            if not isinstance(k, str):
                ok = False
                break
            if vv is None or isinstance(vv, (bool, int, float, str)):
                out[k] = vv
            else:
                ok = False
                break
        return out if ok else repr(v) # pyright: ignore[reportUnknownArgumentType]
    return repr(v) # pyright: ignore[reportUnknownArgumentType]


def _param_meta(p: inspect.Parameter) -> Dict[str, Any]:
    ann = None if p.annotation is _EMPTY else repr(p.annotation)
    return {
        "required": (p.default is _EMPTY),
        "default": jsonish_default(p.default),
        "kind": str(p.kind),        # KEYWORD_ONLY, POSITIONAL_OR_KEYWORD, etc.
        "annotation": ann,
    }


def build_init_fit_options_param_maps(
    cls: Type[Any],
    *,
    fit_include_names: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      {
        "init": { param_name: {required, default, kind, annotation}, ... },
        "fit":  { param_name: {required, default, kind, annotation}, ... },
      }

    Notes:
      - init excludes "self"
      - fit excludes data args by default ("self","Y","T","X","W","Z")
      - You can whitelist fit params via fit_include_names if you want (e.g. {"cache_values","inference"}).
    """
    fit_exclude_names = {"self", "Y", "T", "X", "W", "Z"}

    # ---- __init__ map ----
    init_sig = inspect.signature(cls.__init__)
    init_map: Dict[str, Dict[str, Any]] = {}
    for name, p in init_sig.parameters.items():
        if name == "self":
            continue
        init_map[name] = _param_meta(p)

    # ---- fit map ----
    fit_map: Dict[str, Dict[str, Any]] = {}
    if hasattr(cls, "fit"):
        fit_sig = inspect.signature(cls.fit)  # type: ignore[attr-defined]
        for name, p in fit_sig.parameters.items():
            if name in fit_exclude_names:
                continue
            if fit_include_names is not None and name not in fit_include_names:
                continue
            fit_map[name] = _param_meta(p)

    return {"init": init_map, "fit": fit_map}


def validate_flat_options(
    options: Mapping[str, Any],
    *,
    init_map: Mapping[str, Any],
    fit_map: Mapping[str, Any],
) -> None:
    """
    Validate that flat options keys are known to either __init__ or fit.
    Raise ValueError on unknown keys.
    """
    allowed = set(init_map.keys()) | set(fit_map.keys())
    unknown = [k for k in options.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Unknown option keys: {unknown}")


def split_flat_options(
    options: Mapping[str, Any],
    *,
    init_map: Mapping[str, Any],
    fit_map: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a flat options dict into:
      - init_kwargs (for cls(**init_kwargs))
      - fit_kwargs  (for est.fit(..., **fit_kwargs))
    """
    validate_flat_options(options, init_map=init_map, fit_map=fit_map)
    init_kwargs: Dict[str, Any] = {}
    fit_kwargs: Dict[str, Any] = {}

    for k, v in options.items():
        if k in init_map:
            init_kwargs[k] = v
        else:
            fit_kwargs[k] = v
            
    return init_kwargs, fit_kwargs


def required_init_keys(cls: Type[Any], init_map: Mapping[str, Any]) -> Set[str]:
        sig = inspect.signature(cls.__init__)
        required = {p.name for p in sig.parameters.values() if p.default is p.empty and p.name in init_map}
        return required  


# =============================================================================
# CausalSpec -> columns / arrays (strict to your Pydantic schema)
# =============================================================================

def has_missing(arr: Any) -> bool:
    if arr is None:
        return False
    a = np.asarray(arr)
    try:
        return bool(np.isnan(a).any())
    except Exception:
        return bool(pd.isna(a).any())


def get_input_params_from_spec(
    df: pd.DataFrame,
    specs: CausalSpec, 
    order_X: Optional[List[str]] = None,
    order_W: Optional[List[str]] = None,
    *,
    strict_order: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Returns (y, t, x, w, meta) where x/w are ordered to match the transformer expectations.

    Ordering rules:
      - If oder_X/order_W are provided, they MUST match exactly the X/W columns from specs (strict_order=True).
      - Otherwise we use the specs-defined order.
    """
    y_col = str(specs.Y.column)
    t_col = str(specs.T.column)

    x_cols = specs.X or []
    w_cols = specs.W or []


    # Validate all required columns exist
    validate_columns_exist(df, [y_col, t_col] + (order_X or []) + (order_W or []))

    # Build arrays in the exact order that downstream transformer/model expects
    y: np.ndarray = df[[y_col]].to_numpy()
    t: np.ndarray = df[[t_col]].to_numpy()
    x: Optional[np.ndarray] = df[order_X].to_numpy() if order_X else None
    w: Optional[np.ndarray] = df[order_W].to_numpy() if order_W else None

    # squeeze singleton dims
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]

    overlap_XW = sorted(set(order_X or []).intersection(order_W or []))

    meta: Dict[str, Any] = {
        "y": y_col,
        "t": t_col,
        "x_cols": x_cols,
        "w_cols": w_cols,
        "X_order": order_X,  # <- this is what your ColumnTransformer should use
        "W_order": order_W,  # <- this is what your ColumnTransformer should use
        "overlap_XW": overlap_XW,  # useful for diagnostics if user put same col in both
    }
    return y, t, x, w, meta



def validate_semantic_consistency(spec: CausalSpec, init_kwargs: Mapping[str, Any]) -> None:
    """
    If the user provided options contradict the declared CausalSpec, fail fast.
    Keep it minimal.
    """
    t_kind = getattr(spec.T, "kind", None)
    y_kind = getattr(spec.Y, "kind", None)

    if y_kind == "binary" and "discrete_outcome" in init_kwargs and not bool(init_kwargs["discrete_outcome"]):
        raise ModelSpecError("Spec declares binary outcome but options.discrete_outcome is False.")

    if t_kind in ("binary", "categorical") and "discrete_treatment" in init_kwargs and not bool(init_kwargs["discrete_treatment"]):
        raise ModelSpecError("Spec declares discrete treatment but options.discrete_treatment is False.")

    if t_kind == "categorical":
        baseline = getattr(spec.T, "baseline", None)
        if baseline is not None and "categories" in init_kwargs:
            cats = init_kwargs["categories"]
            if isinstance(cats, list) and cats:
                if cats[0] != baseline:
                    raise ModelSpecError(
                        f"Spec baseline={baseline!r} must be the FIRST category (control). Got categories[0]={cats[0]!r}."
                    )



def serialize_inference_obj(obj: Any) -> Dict[str, Any]:
    # Try common econml inference surfaces; fall back to repr
    if hasattr(obj, "summary_frame"):
        try:
            sf = obj.summary_frame()
            return {"type": "summary_frame", "data": sf.to_dict(orient="list")}
        except Exception:
            pass
    if hasattr(obj, "summary"):
        try:
            s = obj.summary()
            return {"type": "summary", "data": str(s)}
        except Exception:
            pass
    return {"type": "repr", "data": repr(obj)}

def categorical_t0_t1_pairs(spec: CausalSpec) -> Tuple[Any, List[Any]]:
    """
    For categorical treatments:
      - t0 = baseline if provided else levels[0]
      - t1s = all other levels in order

    Returns:
      (t0, t1_list)

    Raises:
      ValueError if spec.T is not categorical or levels invalid.
    """
    t = spec.T
    if getattr(t, "kind", None) != "categorical":
        raise ValueError("categorical_t0_t1_pairs requires spec.T.kind == 'categorical'.")

    levels: List[Any] = list(getattr(t, "levels", None) or [])
    if len(levels) < 2:
        raise ValueError("Categorical treatment requires at least 2 levels.")

    t0 = getattr(t, "baseline", None) if getattr(t, "baseline", None) is not None else levels[0]
    if t0 not in levels:
        raise ValueError(f"baseline {t0!r} must be one of levels={levels!r}.")

    t1s = [lv for lv in levels if lv != t0]
    if not t1s:
        raise ValueError("No target levels found after removing baseline from levels.")

    return t0, t1s


def expand_categorical_contrasts(spec: CausalSpec) -> List[Dict[str, Any]]:
    """
    Returns a list of contrasts suitable for computing ATE per target:
      [{"t0": t0, "t1": t1}, ...]
    """
    t0, t1s = categorical_t0_t1_pairs(spec)
    return [{"t0": t0, "t1": t1} for t1 in t1s]

def materialize_x_query(
    *,
    x_rows: List[Dict[str, Any]],
    x_cols: List[str],
) -> np.ndarray:
    if not x_cols:
        raise ModelSpecError("CATE requires effect modifiers X; x_cols is empty.")

    X_list = []
    x_set = set(x_cols)

    for i, row in enumerate(x_rows):
        row_keys = set(row.keys())
        missing = [c for c in x_cols if c not in row]
        extra = [k for k in row_keys if k not in x_set]
        if missing or extra:
            raise ModelSpecError(
                f"x_rows[{i}] feature mismatch. missing={missing}, extra={extra}. "
                f"Expected exactly: {x_cols}"
            )
        X_list.append([row[c] for c in x_cols]) # pyright: ignore[reportUnknownMemberType]

    return np.asarray(X_list, dtype=float)

def raise_if_x_rows_not_exactly_match_fit_x_cols(
    *,
    x_rows: pd.DataFrame,
    x_cols: List[str],
    require_order: bool = True,
) -> None:
    """
    Enforce x_rows contains ONLY X columns (no extras), and optionally same order.
    """
    # duplicates
    cols = list(x_rows.columns)
    if len(cols) != len(set(cols)):
        dupes = [c for c in set(cols) if cols.count(c) > 1]
        raise ModelSpecError(f"inputs.x_rows has duplicate columns: {dupes}")

    expected = list(x_cols)
    got = cols

    missing = [c for c in expected if c not in set(got)]
    extra = [c for c in got if c not in set(expected)]

    if missing or extra:
        raise ModelSpecError(
            "inputs.x_rows must contain EXACTLY the effect modifier columns spec.X. "
            f"missing={missing}, extra={extra}, expected={expected}, got={got}"
        )

    if require_order and got != expected:
        raise ModelSpecError(
            "inputs.x_rows columns must match spec.X order exactly. "
            f"expected={expected}, got={got}"
        )
        

def validate_columns_exist(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")



#========================================================================
# Missingness allow or not
#=========================================================================
def is_missing_handled(
    *,
    plan: TransformPlan,
    summary: DatasetSummaryModel,
    col_name_list: List[str],
    strict: bool = True,
) -> bool:
    missing_by_col: Dict[str, int] = {p.name: int(p.n_missing) for p in summary.profiles}

    # Index plan by column for O(1) lookup
    plan_by_col = {cp.column: cp for cp in plan.columns}

    forbidden: List[str] = []
    needs_allow_missing: List[str] = []

    for col in col_name_list:
        col_plan = plan_by_col.get(col)
        if col_plan is None:
            if strict:
                raise ValueError(f"Column '{col}' is in col_name_list but not present in TransformPlan.")
            continue

        enc = col_plan.encoding

        if enc.preset == "drop":
            continue  # dropped => irrelevant downstream

        n_missing = missing_by_col.get(col)
        if n_missing is None:
            if strict:
                raise ValueError(f"Column '{col}' is in col_name_list but not present in DatasetSummaryModel.")
            continue

        if n_missing <= 0:
            continue

        status = _missingness_handling(enc)
        if status == "FORBIDS":
            forbidden.append(col)
        elif status == "UNHANDLED":
            needs_allow_missing.append(col)

    if forbidden and strict:
        raise ValueError(
            "Missingness is present in columns that are configured to forbid missing values: "
            + ", ".join(sorted(forbidden))
            + ". Fix by changing encoding missing=..., imputing upstream, or dropping the column."
        )

    return bool(needs_allow_missing)


def _missingness_handling(enc: EncodingPresetSpec) -> str:
    """
    Returns one of: "HANDLED", "UNHANDLED", "FORBIDS"
    """
    # Structural
    if isinstance(enc, DropParams):
        return "HANDLED"  # ignored upstream
    if isinstance(enc, PassthroughParams):
        return "UNHANDLED"  # NaNs pass straight through

    # Categorical
    if isinstance(enc, CatOneHotParams):
        # - impute_token: explicit fill_value -> no NaNs
        # - dummy_na: missing represented as its own category -> no NaNs (conceptually)
        # - error: pipeline should fail if missing exists
        if enc.missing in ("impute_token", "dummy_na"):
            return "HANDLED"
        return "FORBIDS"  # missing == "error"

    # Numeric: all your Num* presets explicitly impute -> handled
    if isinstance(enc, (NumStandardParams, NumMinMaxParams, NumLog1pParams)):
        return "HANDLED"

    # Datetime epoch seconds:
    # Your params include errors/coerce and add_missing_indicator, but no explicit imputation knob.
    # That means NaNs can still exist after conversion -> treat as UNHANDLED.
    if isinstance(enc, DateTimeEpochParams):
        return "UNHANDLED"

    # Explicit mapping: handled unless missing='error'
    if isinstance(enc, MapBinaryParams):
        if enc.missing == "error":
            return "FORBIDS"
        return "HANDLED"

    if isinstance(enc, MapOrdinalParams): # pyright: ignore[reportUnnecessaryIsInstance]
        if enc.missing == "error":
            return "FORBIDS"
        return "HANDLED"

    # Should be unreachable if EncodingPresetSpec is exhaustive
    return "UNHANDLED"

    