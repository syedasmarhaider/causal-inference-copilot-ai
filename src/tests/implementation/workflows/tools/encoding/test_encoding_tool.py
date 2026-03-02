import types

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# ✅ ADJUST THIS IMPORT PATH IF YOUR FILE NAME DIFFERS
from python.implementation.workflows.tools.encoding.encoding_tool import (  # noqa: E501
    BinaryMapTransformer,
    DateTimeToEpochSecondsTransformer,
    EncodingTool,
    Log1pSafeTransformer,
    MinMaxEpsScaler,
    OrdinalMapTransformer,
    RaiseIfMissing,
    _require_non_empty, # pyright: ignore[reportPrivateUsage]
    compile_plan_to_transformers,
)

from python.implementation.workflows.tools.common.model.encoding_plan import TransformPlan
import python.implementation.workflows.tools.encoding.encoding_tool as m


def _mk_plan(*cols: dict) -> TransformPlan:
    # cols entries: {"column": "...", "role": "X|W", "encoding": {...}}
    return TransformPlan(columns=[c for c in cols])


def test_require_non_empty_errors():
    _require_non_empty("X_order", ["a"])  # ok
    with pytest.raises(ValueError, match="must be provided"):
        _require_non_empty("X_order", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be non-empty"):
        _require_non_empty("X_order", [])


def test_encoding_tool_compile_happy_path_dense():
    plan = _mk_plan(
        {
            "column": "age",
            "role": "X",
            "encoding": {"preset": "num_standard", "impute": "median", "add_missing_indicator": True},
        },
        {
            "column": "gender",
            "role": "W",
            "encoding": {
                "preset": "cat_onehot",
                "missing": "impute_token",
                "missing_token": "__MISSING__",
                "drop_first": False,
                "handle_unknown": "ignore",
            },
        },
    )

    tool = EncodingTool()
    compiled = tool.compile(plan, X_order=["age"], W_order=["gender"], dense_output=True)

    assert compiled.X_order == ("age",)
    assert compiled.W_order == ("gender",)

    # Use DataFrame to preserve dtypes while still using integer indices inside ColumnTransformer.
    df_x = pd.DataFrame({"age": [1.0, np.nan, 3.0]})
    df_xw = pd.DataFrame({"age": [1.0, np.nan, 3.0], "gender": ["M", None, "F"]})

    out_x = compiled.pre_X.fit_transform(df_x)
    assert isinstance(out_x, np.ndarray)
    assert out_x.shape[0] == 3
    assert out_x.shape[1] == 2  # value + missing indicator

    out_xw = compiled.pre_XW.fit_transform(df_xw)
    assert isinstance(out_xw, np.ndarray)
    assert out_xw.shape[0] == 3
    # age -> 2 cols, gender -> {F,M,__MISSING__} => 3 cols => total 5
    assert out_xw.shape[1] == 5

    # feature names should be available after fit
    names = compiled.pre_XW.get_feature_names_out()
    assert len(names) == out_xw.shape[1]


def test_compile_require_full_coverage_missing_plan_raises():
    plan = _mk_plan(
        {"column": "age", "role": "X", "encoding": {"preset": "num_standard"}},
        {"column": "gender", "role": "W", "encoding": {"preset": "cat_onehot"}},
    )

    with pytest.raises(ValueError, match="Missing encoding plan"):
        compile_plan_to_transformers(
            plan,
            X_order=["age"],
            W_order=["gender", "city"],  # city missing from plan
            dense_output=True,
            require_full_coverage=True,
        )


def test_compile_require_full_coverage_extra_plan_raises():
    plan = _mk_plan(
        {"column": "age", "role": "X", "encoding": {"preset": "num_standard"}},
        {"column": "gender", "role": "W", "encoding": {"preset": "cat_onehot"}},
        {"column": "extra", "role": "W", "encoding": {"preset": "drop"}},
    )

    with pytest.raises(ValueError, match="Plan contains columns not present"):
        compile_plan_to_transformers(
            plan,
            X_order=["age"],
            W_order=["gender"],
            dense_output=True,
            require_full_coverage=True,
        )


def test_compile_duplicate_column_plans_raises():
    # Duplicate is blocked both by TransformPlan validator and by compile()’s own sanity check.
    with pytest.raises(ValidationError):
        _mk_plan(
            {"column": "age", "role": "X", "encoding": {"preset": "num_standard"}},
            {"column": "age", "role": "W", "encoding": {"preset": "drop"}},
        )


def test_compile_role_mismatch_raises():
    plan = _mk_plan(
        # ❌ intentionally wrong: age is in X_order but role is W
        {"column": "age", "role": "W", "encoding": {"preset": "num_standard"}},
        # ✅ satisfy TransformPlan invariant: at least one X exists
        {"column": "x_ok", "role": "X", "encoding": {"preset": "num_standard"}},
        # ✅ W exists
        {"column": "gender", "role": "W", "encoding": {"preset": "cat_onehot"}},
    )

    with pytest.raises(ValueError, match="is in X_order but plan role"):
        compile_plan_to_transformers(
            plan,
            X_order=["age", "x_ok"],
            W_order=["gender"],
            dense_output=True,
        )


def test_compile_requires_X_and_W_present_in_plan():
    with pytest.raises(ValidationError, match="must contain at least one X"):
        _mk_plan({"column": "w1", "role": "W", "encoding": {"preset": "drop"}})

    with pytest.raises(ValidationError, match="must contain at least one W"):
        _mk_plan({"column": "x1", "role": "X", "encoding": {"preset": "drop"}})


def test_compile_all_X_dropped_raises():
    plan = _mk_plan(
        {"column": "age", "role": "X", "encoding": {"preset": "drop"}},
        {"column": "gender", "role": "W", "encoding": {"preset": "cat_onehot"}},
    )

    with pytest.raises(ValueError, match="All X columns are dropped"):
        compile_plan_to_transformers(plan, X_order=["age"], W_order=["gender"], dense_output=True)


def test_compile_unsupported_preset_raises():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "passthrough"}},
        {"column": "w", "role": "W", "encoding": {"preset": "passthrough"}},
    )
    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    # Force-hit the "Unsupported preset" branch by calling the internal compiler via a fake plan:
    fake_plan = types.SimpleNamespace(
        columns=[
            types.SimpleNamespace(column="x", role="X", encoding=types.SimpleNamespace(preset="bogus")),
            types.SimpleNamespace(column="w", role="W", encoding=types.SimpleNamespace(preset="passthrough")),
        ]
    )
    with pytest.raises(ValueError, match="Unsupported preset"):
        compile_plan_to_transformers(  # type: ignore[arg-type]
            fake_plan,
            X_order=["x"],
            W_order=["w"],
            dense_output=True,
            require_full_coverage=False,
        )


def test_make_one_hot_encoder_backward_compat_monkeypatch(monkeypatch: pytest.MonkeyPatch):
    # Hit the except TypeError path (older sklearn API) by replacing OneHotEncoder with a stub.


    class StubOHE:
        def __init__(self, **kwargs):
            if "sparse_output" in kwargs:
                raise TypeError("no sparse_output in this stub")
            self.kwargs = kwargs

    monkeypatch.setattr(m, "OneHotEncoder", StubOHE)
    enc = m._make_one_hot_encoder(
        handle_unknown="ignore",
        drop_first=False,
        max_categories=None,
        dense_output=False,
    )
    assert isinstance(enc, StubOHE)
    assert enc.kwargs["sparse"] is True


def test_compile_dense_output_false_returns_sparse():
    # With dense_output=False, encoder emits sparse and ColumnTransformer has sparse_threshold=1.0
    plan = _mk_plan(
        {"column": "xcat", "role": "X", "encoding": {"preset": "cat_onehot", "missing": "impute_token"}},
        {"column": "wcat", "role": "W", "encoding": {"preset": "cat_onehot", "missing": "impute_token"}},
    )
    compiled = compile_plan_to_transformers(plan, X_order=["xcat"], W_order=["wcat"], dense_output=False)

    df = pd.DataFrame({"xcat": ["a", "b", "c"], "wcat": ["u", "v", "w"]})
    out = compiled.pre_XW.fit_transform(df)

    # Avoid hard dependency on scipy API details; sparse matrices have `toarray`.
    assert hasattr(out, "toarray")


def test_raise_if_missing_fit_transform_and_names():
    t = RaiseIfMissing()

    ok = np.array([["x"], ["y"]], dtype=object)
    t.fit(ok)
    out = t.transform(ok)
    assert out.shape == ok.shape

    bad = np.array([[None], ["x"]], dtype=object)
    with pytest.raises(ValueError, match="Missing values found"):
        t.fit(bad)
    with pytest.raises(ValueError, match="Missing values found"):
        t.transform(bad)

    assert t.get_feature_names_out(["col"]).tolist() == ["col"]
    assert t.get_feature_names_out().tolist() == ["feature"]


def test_log1p_safe_transformer_branches_and_names():
    t = Log1pSafeTransformer(allow_negative=False)

    with pytest.raises(ValueError, match="expected shape"):
        t.transform(np.array([1.0, 2.0]))  # not 2D (n,1)

    with pytest.raises(ValueError, match="<= -1"):
        t.transform(np.array([[-1.0], [0.0]]))

    with pytest.raises(ValueError, match="negative values found"):
        t.transform(np.array([[-0.5], [0.0]]))

    t2 = Log1pSafeTransformer(allow_negative=True)
    out = t2.transform(np.array([[-0.5], [0.0], [1.0]]))
    assert out.shape == (3, 1)

    assert t2.get_feature_names_out(["x"]).tolist() == ["x"]
    assert t2.get_feature_names_out().tolist() == ["log1p"]


def test_minmax_eps_scaler_constant_col_and_names():
    scaler = MinMaxEpsScaler(eps=1e-12)

    X = np.array([[5.0], [5.0], [5.0]])
    scaler.fit(X)
    out = scaler.transform(X)
    assert out.shape == (3, 1)
    assert np.allclose(out, 0.0)

    with pytest.raises(ValueError, match="not fitted"):
        MinMaxEpsScaler().transform(X)

    assert scaler.get_feature_names_out(["x"]).tolist() == ["x"]


def test_binary_map_transformer_branches():
    # unknown handling + missing as_unknown
    t = BinaryMapTransformer(
        mapping={"yes": 1.0, "no": 0.0},
        allow_unknown=True,
        unknown_value=-9.0,
        missing="as_unknown",
        missing_token=None,
    )
    t.fit(np.array([["yes"]], dtype=object))
    out = t.transform(np.array([["yes"], ["maybe"], [None]], dtype=object))
    assert out.shape == (3, 1)
    assert out[0, 0] == 1.0
    assert out[1, 0] == -9.0  # unknown -> unknown_value
    assert out[2, 0] == -9.0  # missing -> unknown_value

    # missing='error'
    t_err = BinaryMapTransformer(
        mapping={"yes": 1.0, "no": 0.0},
        allow_unknown=True,
        unknown_value=-1.0,
        missing="error",
        missing_token=None,
    )
    t_err.fit(np.array([["yes"]], dtype=object))
    with pytest.raises(ValueError, match="missing values found"):
        t_err.transform(np.array([[None]], dtype=object))

    # allow_unknown=False should raise on unknown categories
    t_strict = BinaryMapTransformer(
        mapping={"yes": 1.0, "no": 0.0},
        allow_unknown=False,
        unknown_value=None,
        missing="impute_token",
        missing_token="no",
    )
    t_strict.fit(np.array([["yes"]], dtype=object))
    with pytest.raises(ValueError, match="unknown categories"):
        t_strict.transform(np.array([["maybe"]], dtype=object))

    assert t.get_feature_names_out(["b"]).tolist() == ["b"]
    assert t.get_feature_names_out().tolist() == ["map_binary"]


def test_ordinal_map_transformer_impute_token_prepend_append_and_unknown():
    # prepend token
    t_pre = OrdinalMapTransformer(
        order=["low", "mid", "high"],
        start=10,
        allow_unknown=True,
        unknown_value=-1,
        missing="impute_token",
        missing_token="__NA__",
        token_position="prepend",
    )
    t_pre.fit(np.array([["low"]], dtype=object))
    out = t_pre.transform(np.array([[None], ["low"], ["high"], ["zzz"]], dtype=object))
    assert out.shape == (4, 1)
    assert out[0, 0] == 10.0  # __NA__ inserted at start => start+0
    assert out[1, 0] == 11.0  # low now index 1
    assert out[2, 0] == 13.0  # high now index 3
    assert out[3, 0] == -1.0  # unknown

    # append token
    t_app = OrdinalMapTransformer(
        order=["a", "b"],
        start=0,
        allow_unknown=False,
        unknown_value=None,
        missing="impute_token",
        missing_token="__NA__",
        token_position="append",
    )
    t_app.fit(np.array([["a"]], dtype=object))
    out2 = t_app.transform(np.array([[None], ["a"], ["b"]], dtype=object))
    assert out2[:, 0].tolist() == [2.0, 0.0, 1.0]  # __NA__ appended => index 2

    # missing='error' branch
    t_err = OrdinalMapTransformer(
        order=["a"],
        start=0,
        allow_unknown=True,
        unknown_value=-9,
        missing="error",
        missing_token=None,
        token_position=None,
    )
    t_err.fit(np.array([["a"]], dtype=object))
    with pytest.raises(ValueError, match="missing values found"):
        t_err.transform(np.array([[None]], dtype=object))

    assert t_pre.get_feature_names_out(["ord"]).tolist() == ["ord"]
    assert t_pre.get_feature_names_out().tolist() == ["map_ordinal"]


def test_datetime_to_epoch_seconds_branches_and_names():
    # errors='raise' should throw on invalid (non-null) strings
    t_raise = DateTimeToEpochSecondsTransformer(errors="raise", unit="s", add_missing_indicator=False)
    t_raise.fit(np.array([["2020-01-01"]], dtype=object))
    with pytest.raises(ValueError, match="unparseable"):
        t_raise.transform(np.array([["not-a-date"]], dtype=object))

    # missing indicator + tz-aware conversion path
    t = DateTimeToEpochSecondsTransformer(errors="coerce", unit="s", add_missing_indicator=True)
    out = t.transform(np.array([["2020-01-01T00:00:00+01:00"], [None]], dtype=object))
    assert out.shape == (2, 2)  # value + missing indicator
    assert out[1, 1] == 1  # missing row -> indicator 1

    assert t.get_feature_names_out(["dt"]).tolist() == ["dt", "dt_missing"]
    t2 = DateTimeToEpochSecondsTransformer(errors="coerce", unit="s", add_missing_indicator=False)
    assert t2.get_feature_names_out(["dt"]).tolist() == ["dt"]
    
    
    # -----------------------------------------------------------------------------
# Compile-level behavioral edge cases
# -----------------------------------------------------------------------------
def test_compile_require_full_coverage_false_allows_partial_plan():
    # Contract: when coverage is not required, compiler should accept plan that
    # doesn't mention every X/W column (and should compile what exists).
    plan = _mk_plan(
        {"column": "x1", "role": "X", "encoding": {"preset": "num_standard"}},
        {"column": "w1", "role": "W", "encoding": {"preset": "cat_onehot"}},
    )

    compiled = compile_plan_to_transformers(
        plan,
        X_order=["x1", "x2"],      # x2 not in plan
        W_order=["w1", "w2"],      # w2 not in plan
        dense_output=True,
        require_full_coverage=False,
    )

    # Should still compile, and should be usable on arrays that contain the full X|W layout.
    XW = pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [10.0, 20.0],      # no plan
            "w1": ["a", "b"],
            "w2": ["u", "v"],        # no plan
        }
    )
    out = compiled.pre_XW.fit_transform(XW)
    assert out.shape[0] == 2


def test_compiled_transformers_work_with_numpy_arrays_not_only_dataframes():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "num_standard"}},
        {"column": "w", "role": "W", "encoding": {"preset": "cat_onehot", "missing": "impute_token"}},
    )
    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    X_np = np.array([[1.0], [np.nan], [3.0]], dtype=object)
    XW_np = np.array([[1.0, "a"], [np.nan, None], [3.0, "b"]], dtype=object)

    out_x = compiled.pre_X.fit_transform(X_np)
    out_xw = compiled.pre_XW.fit_transform(XW_np)

    assert isinstance(out_x, np.ndarray)
    assert isinstance(out_xw, np.ndarray)
    assert out_x.shape[0] == 3
    assert out_xw.shape[0] == 3


# -----------------------------------------------------------------------------
# cat_onehot behavioral edges (unknown categories + missing policies)
# -----------------------------------------------------------------------------
def test_cat_onehot_handle_unknown_error_raises_on_unseen_category():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "passthrough"}},
        {
            "column": "w",
            "role": "W",
            "encoding": {
                "preset": "cat_onehot",
                "handle_unknown": "error",
                "missing": "impute_token",
                "missing_token": "__MISSING__",
            },
        },
    )

    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    train = pd.DataFrame({"x": [0, 1], "w": ["A", "A"]})
    test = pd.DataFrame({"x": [0], "w": ["B"]})  # unseen

    compiled.pre_XW.fit(train)
    with pytest.raises(Exception):  # sklearn message differs by version
        compiled.pre_XW.transform(test)


def test_cat_onehot_handle_unknown_ignore_does_not_raise_on_unseen_category():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "passthrough"}},
        {
            "column": "w",
            "role": "W",
            "encoding": {
                "preset": "cat_onehot",
                "handle_unknown": "ignore",
                "missing": "impute_token",
                "missing_token": "__MISSING__",
            },
        },
    )

    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    train = pd.DataFrame({"x": [0, 1], "w": ["A", "A"]})
    test = pd.DataFrame({"x": [0], "w": ["B"]})  # unseen

    compiled.pre_XW.fit(train)
    out = compiled.pre_XW.transform(test)
    assert out.shape[0] == 1


def test_cat_onehot_missing_error_raises_on_missing_values():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "passthrough"}},
        {"column": "w", "role": "W", "encoding": {"preset": "cat_onehot", "missing": "error"}},
    )
    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    df = pd.DataFrame({"x": [0, 1], "w": ["A", None]})
    with pytest.raises(ValueError, match="Missing values found"):
        compiled.pre_XW.fit_transform(df)


def test_cat_onehot_dummy_na_treats_missing_as_category():
    plan = _mk_plan(
        {"column": "x", "role": "X", "encoding": {"preset": "passthrough"}},
        {"column": "w", "role": "W", "encoding": {"preset": "cat_onehot", "missing": "dummy_na"}},
    )
    compiled = compile_plan_to_transformers(plan, X_order=["x"], W_order=["w"], dense_output=True)

    df = pd.DataFrame({"x": [0, 1, 2], "w": ["A", None, "A"]})
    out = compiled.pre_XW.fit_transform(df)
    assert out.shape[0] == 3


# -----------------------------------------------------------------------------
# datetime_epoch_seconds: invalid unit + unit scaling
# -----------------------------------------------------------------------------
def test_datetime_epoch_seconds_invalid_unit_raises_in_fit():
    t = DateTimeToEpochSecondsTransformer(errors="coerce", unit="minutes", add_missing_indicator=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid unit"):
        t.fit(np.array([["2020-01-01"]], dtype=object))


def test_datetime_epoch_seconds_unit_scaling_s_vs_ms():
    dt = np.array([["1970-01-01T00:00:01Z"]], dtype=object)

    t_s = DateTimeToEpochSecondsTransformer(errors="coerce", unit="s", add_missing_indicator=False)
    t_ms = DateTimeToEpochSecondsTransformer(errors="coerce", unit="ms", add_missing_indicator=False)

    out_s = t_s.transform(dt)[0, 0]
    out_ms = t_ms.transform(dt)[0, 0]

    # 1 second == 1000 milliseconds
    assert out_ms == pytest.approx(out_s * 1000.0)


# -----------------------------------------------------------------------------
# map_binary: fit-time guards + transform unknowns
# -----------------------------------------------------------------------------
def test_map_binary_fit_rejects_empty_mapping():
    t = BinaryMapTransformer(
        mapping={},
        allow_unknown=True,
        unknown_value=-1.0,
        missing="as_unknown",
        missing_token=None,
    )
    with pytest.raises(ValueError, match="mapping must be non-empty"):
        t.fit(np.array([["x"]], dtype=object))


def test_map_binary_fit_rejects_missing_token_not_in_mapping():
    t = BinaryMapTransformer(
        mapping={"yes": 1.0, "no": 0.0},
        allow_unknown=True,
        unknown_value=-1.0,
        missing="impute_token",
        missing_token="__NA__",  # not in mapping
    )
    with pytest.raises(ValueError, match="missing_token must exist in mapping"):
        t.fit(np.array([["yes"]], dtype=object))


def test_map_binary_fit_rejects_unknown_value_when_allow_unknown_false():
    t = BinaryMapTransformer(
        mapping={"yes": 1.0, "no": 0.0},
        allow_unknown=False,
        unknown_value=-9.0,  # forbidden in strict mode
        missing="error",
        missing_token=None,
    )
    with pytest.raises(ValueError, match="unknown_value must be null"):
        t.fit(np.array([["yes"]], dtype=object))


# -----------------------------------------------------------------------------
# map_ordinal: fit-time duplicate protection + strict unknowns
# -----------------------------------------------------------------------------
def test_map_ordinal_fit_rejects_duplicate_order():
    t = OrdinalMapTransformer(
        order=["a", "a"],
        start=0,
        allow_unknown=True,
        unknown_value=-1,
        missing="error",
        missing_token=None,
        token_position=None,
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        t.fit(np.array([["a"]], dtype=object))


def test_map_ordinal_strict_unknown_raises_on_unseen_category():
    t = OrdinalMapTransformer(
        order=["low", "high"],
        start=0,
        allow_unknown=False,
        unknown_value=None,
        missing="impute_token",
        missing_token="low",
        token_position="prepend",
    )
    t.fit(np.array([["low"]], dtype=object))

    with pytest.raises(ValueError, match="unknown categories"):
        t.transform(np.array([["mid"]], dtype=object))