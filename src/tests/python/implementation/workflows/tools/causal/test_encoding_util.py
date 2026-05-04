from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_util import (
    BinaryMapTransformer,
    DateTimeToEpochSecondsTransformer,
    Log1pSafeTransformer,
    MinMaxEpsScaler,
    OrdinalMapTransformer,
    RaiseIfMissing,
    compile_plan_to_transformers,
)


def _plan(*columns: dict[str, object]) -> TransformPlan:
    return TransformPlan.model_validate({"columns": list(columns)})


def test_compile_allows_empty_effect_modifiers_when_covariates_exist() -> None:
    plan = _plan(
        {
            "column": "age",
            "role": "covariate",
            "encoding": {"preset": "num_standard"},
        }
    )

    compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=[],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
    )

    assert compiled.pre_X is None
    assert compiled.pre_XW is not None


def test_compile_respects_requested_feature_order_even_when_plan_order_differs() -> None:
    plan = _plan(
        {
            "column": "age",
            "role": "covariate",
            "encoding": {"preset": "passthrough"},
        },
        {
            "column": "segment",
            "role": "effect_modifier",
            "encoding": {
                "preset": "map_ordinal",
                "order": ["A", "B"],
                "allow_unknown": True,
                "unknown_value": -1,
            },
        },
    )

    compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=["segment"],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
    )

    x = np.asarray([["B"], ["A"]], dtype=object)
    xw = np.asarray([["B", 10.0], ["A", 20.0]], dtype=object)

    x_tx = compiled.pre_X.fit_transform(x)  # type: ignore[union-attr]
    xw_tx = compiled.pre_XW.fit_transform(xw)

    assert x_tx[:, 0].tolist() == [1.0, 0.0]
    assert xw_tx[:, 0].tolist() == [1.0, 0.0]
    assert xw_tx[:, 1].tolist() == [10.0, 20.0]
    feature_names = list(compiled.pre_XW.get_feature_names_out())
    assert feature_names == [
        "segment__x0",
        "age__x1",
    ]


def test_compile_drop_first_effect_modifier_onehot_only_changes_pre_x() -> None:
    plan = _plan(
        {
            "column": "segment",
            "role": "effect_modifier",
            "encoding": {"preset": "cat_onehot", "handle_unknown": "ignore"},
        },
        {
            "column": "age",
            "role": "covariate",
            "encoding": {"preset": "passthrough"},
        },
    )
    x = np.asarray([["A"], ["B"], ["A"]], dtype=object)
    xw = np.asarray([["A", 10.0], ["B", 20.0], ["A", 30.0]], dtype=object)

    default_compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=["segment"],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
    )
    default_x = default_compiled.pre_X.fit_transform(x)  # type: ignore[union-attr]

    drop_first_compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=["segment"],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
        drop_first_effect_modifier_onehot=True,
    )
    drop_first_x = drop_first_compiled.pre_X.fit_transform(x)  # type: ignore[union-attr]
    drop_first_xw = drop_first_compiled.pre_XW.fit_transform(xw)

    assert default_x.shape == (3, 2)
    assert drop_first_x.shape == (3, 1)
    assert drop_first_xw.shape == (3, 3)
    assert plan.columns[0].encoding.drop_first is False  # type: ignore[attr-defined]


def test_compile_rejects_all_dropped_active_view() -> None:
    plan = _plan(
        {
            "column": "age",
            "role": "covariate",
            "encoding": {"preset": "drop"},
        }
    )

    with pytest.raises(
        ValueError,
        match=r"All effect_modifier/covariate columns are dropped",
    ):
        compile_plan_to_transformers(
            plan=plan,
            effect_modifiers=[],
            covariates=["age"],
            dense_output=True,
            require_full_coverage=True,
        )


def test_compile_rejects_all_dropped_effect_modifier_view_when_present() -> None:
    plan = _plan(
        {
            "column": "segment",
            "role": "effect_modifier",
            "encoding": {"preset": "drop"},
        },
        {
            "column": "age",
            "role": "covariate",
            "encoding": {"preset": "passthrough"},
        },
    )

    with pytest.raises(
        ValueError,
        match=r"All effect_modifier columns are dropped",
    ):
        compile_plan_to_transformers(
            plan=plan,
            effect_modifiers=["segment"],
            covariates=["age"],
            dense_output=True,
            require_full_coverage=True,
        )


def test_compile_and_fit_transform_small_mixed_plan() -> None:
    plan = _plan(
        {
            "column": "segment",
            "role": "effect_modifier",
            "encoding": {"preset": "cat_onehot", "handle_unknown": "ignore"},
        },
        {
            "column": "visit_time",
            "role": "effect_modifier",
            "encoding": {
                "preset": "datetime_epoch_seconds",
                "errors": "coerce",
                "unit": "s",
                "add_missing_indicator": True,
            },
        },
        {
            "column": "age",
            "role": "covariate",
            "encoding": {
                "preset": "num_standard",
                "impute": "median",
                "add_missing_indicator": True,
            },
        },
        {
            "column": "flag",
            "role": "covariate",
            "encoding": {
                "preset": "map_binary",
                "mapping": {"Y": 1.0, "N": 0.0},
                "allow_unknown": True,
                "unknown_value": -1.0,
            },
        },
    )

    compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=["segment", "visit_time"],
        covariates=["age", "flag"],
        dense_output=True,
        require_full_coverage=True,
    )

    x = np.asarray(
        [
            ["A", "2026-01-01T00:00:00"],
            ["B", None],
            ["A", "2026-01-03T00:00:00"],
        ],
        dtype=object,
    )
    xw = np.asarray(
        [
            ["A", "2026-01-01T00:00:00", 10.0, "Y"],
            ["B", None, np.nan, None],
            ["A", "2026-01-03T00:00:00", 30.0, "N"],
        ],
        dtype=object,
    )

    x_tx = compiled.pre_X.fit_transform(x)  # type: ignore[union-attr]
    xw_tx = compiled.pre_XW.fit_transform(xw)

    assert x_tx.shape == (3, 4)
    assert xw_tx.shape == (3, 7)
    assert len(compiled.pre_X.get_feature_names_out()) == 4  # type: ignore[union-attr]
    assert len(compiled.pre_XW.get_feature_names_out()) == 7


def test_raise_if_missing_passes_clean_input_and_fails_on_missing() -> None:
    transformer = RaiseIfMissing()

    clean = np.asarray([["A"], ["B"]], dtype=object)
    assert transformer.fit(clean).transform(clean).tolist() == clean.tolist()

    with pytest.raises(ValueError, match=r"Missing values found"):
        transformer.fit(np.asarray([["A"], [None]], dtype=object))


def test_log1p_safe_transformer_validates_shape_and_values() -> None:
    transformer = Log1pSafeTransformer(allow_negative=False)

    out = transformer.fit_transform(np.asarray([[0.0], [3.0]], dtype=float))
    assert_allclose(out[:, 0], np.log1p([0.0, 3.0]))

    with pytest.raises(ValueError, match=r"expected shape"):
        transformer.transform(np.asarray([1.0, 2.0], dtype=float))

    with pytest.raises(ValueError, match=r"values <= -1"):
        transformer.transform(np.asarray([[-1.0], [0.0]], dtype=float))

    with pytest.raises(ValueError, match=r"allow_negative=False"):
        transformer.transform(np.asarray([[-0.5], [0.0]], dtype=float))


def test_minmax_eps_scaler_handles_constant_columns_and_requires_fit() -> None:
    scaler = MinMaxEpsScaler(eps=1e-12)

    with pytest.raises(ValueError, match=r"not fitted"):
        scaler.transform(np.asarray([[5.0], [5.0]], dtype=float))

    out = scaler.fit_transform(np.asarray([[5.0], [5.0]], dtype=float))
    assert_allclose(out, np.asarray([[0.0], [0.0]], dtype=float))


def test_binary_map_transformer_handles_unknown_and_missing_modes() -> None:
    transformer = BinaryMapTransformer(
        mapping={"Y": 1.0, "N": 0.0},
        allow_unknown=True,
        unknown_value=-1.0,
        missing="as_unknown",
        missing_token=None,
    )
    transformer.fit(np.asarray([["Y"], ["N"]], dtype=object))
    out = transformer.transform(np.asarray([["Y"], ["U"], [None]], dtype=object))
    assert out[:, 0].tolist() == [1.0, -1.0, -1.0]

    missing_token_transformer = BinaryMapTransformer(
        mapping={"Y": 1.0, "N": 0.0, "__MISSING__": 0.5},
        allow_unknown=True,
        unknown_value=-1.0,
        missing="impute_token",
        missing_token="__MISSING__",
    )
    missing_token_transformer.fit(np.asarray([["Y"], ["N"]], dtype=object))
    out2 = missing_token_transformer.transform(np.asarray([[None], ["N"]], dtype=object))
    assert out2[:, 0].tolist() == [0.5, 0.0]

    error_transformer = BinaryMapTransformer(
        mapping={"Y": 1.0, "N": 0.0},
        allow_unknown=False,
        unknown_value=None,
        missing="error",
        missing_token=None,
    )
    error_transformer.fit(np.asarray([["Y"], ["N"]], dtype=object))
    with pytest.raises(ValueError, match=r"missing='error'"):
        error_transformer.transform(np.asarray([[None]], dtype=object))

    with pytest.raises(ValueError, match=r"unknown categories found"):
        error_transformer.transform(np.asarray([["U"]], dtype=object))


@pytest.mark.parametrize(
    ("token_position", "expected"),
    [
        ("prepend", [0.0, 1.0, 2.0]),
        ("append", [2.0, 0.0, 1.0]),
    ],
)
def test_ordinal_map_transformer_handles_token_insertion(
    token_position: str, expected: list[float]
) -> None:
    transformer = OrdinalMapTransformer(
        order=["low", "high"],
        start=0,
        allow_unknown=True,
        unknown_value=-1,
        missing="impute_token",
        missing_token="__M__",
        token_position=token_position,  # type: ignore[arg-type]
    )
    transformer.fit(np.asarray([["low"], ["high"]], dtype=object))
    out = transformer.transform(np.asarray([[None], ["low"], ["high"]], dtype=object))
    assert out[:, 0].tolist() == expected


def test_ordinal_map_transformer_rejects_duplicates_and_unknowns() -> None:
    with pytest.raises(ValueError, match=r"must not contain duplicates"):
        OrdinalMapTransformer(
            order=["low", "low"],
            start=0,
            allow_unknown=True,
            unknown_value=-1,
            missing="as_unknown",
            missing_token=None,
            token_position=None,
        ).fit(np.asarray([["low"]], dtype=object))

    transformer = OrdinalMapTransformer(
        order=["low", "high"],
        start=0,
        allow_unknown=False,
        unknown_value=None,
        missing="as_unknown",
        missing_token=None,
        token_position=None,
    )
    transformer.fit(np.asarray([["low"], ["high"]], dtype=object))
    with pytest.raises(ValueError, match=r"unknown categories found"):
        transformer.transform(np.asarray([["mid"]], dtype=object))


def test_datetime_to_epoch_seconds_handles_parse_errors_missing_indicator_and_mixed_timezones() -> (
    None
):
    transformer = DateTimeToEpochSecondsTransformer(
        errors="coerce",
        unit="s",
        add_missing_indicator=True,
    )
    out = transformer.fit_transform(np.asarray([["2026-01-01T00:00:00"], [None]], dtype=object))
    assert out.shape == (2, 2)
    assert out[1, 1] == 1.0
    assert out[0, 1] == 0.0

    raise_transformer = DateTimeToEpochSecondsTransformer(
        errors="raise",
        unit="s",
        add_missing_indicator=False,
    )
    with pytest.raises(ValueError, match=r"unparseable values found"):
        raise_transformer.transform(np.asarray([["bad-datetime"]], dtype=object))

    tz_transformer = DateTimeToEpochSecondsTransformer(
        errors="coerce",
        unit="s",
        add_missing_indicator=False,
    )
    tz_out = tz_transformer.transform(
        np.asarray(
            [
                ["2026-01-01T00:00:00+00:00"],
                ["2026-01-01T01:00:00+01:00"],
            ],
            dtype=object,
        )
    )
    assert tz_out.shape == (2, 1)
    assert tz_out[0, 0] == tz_out[1, 0]


@pytest.mark.parametrize("then_scale", ["none", "standard", "minmax"])
def test_num_log1p_branch_handles_missing_indicator_without_shape_failure(then_scale: str) -> None:
    plan = _plan(
        {
            "column": "age",
            "role": "covariate",
            "encoding": {
                "preset": "num_log1p",
                "impute": "median",
                "add_missing_indicator": True,
                "allow_negative": False,
                "then_scale": then_scale,
            },
        }
    )

    compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=[],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
    )

    out = compiled.pre_XW.fit_transform(np.asarray([[1.0], [np.nan], [3.0]], dtype=float))

    assert out.shape == (3, 2)
    assert_allclose(out[:, 1], np.asarray([0.0, 1.0, 0.0]))
    assert np.isfinite(out[:, 0]).all()

    if then_scale == "none":
        assert_allclose(out[:, 0], np.log1p([1.0, 2.0, 3.0]))
