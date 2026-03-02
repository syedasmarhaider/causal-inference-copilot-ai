import pytest
from pydantic import TypeAdapter, ValidationError

from python.implementation.workflows.tools.common.model.encoding_plan import (
    EncodingPresetSpec,
    TransformPlan,
)

ENC = TypeAdapter(EncodingPresetSpec)


def parse_enc(payload: dict):
    # Validates the discriminated union (EncodingPresetSpec)
    return ENC.validate_python(payload)


def test_transform_plan_duplicate_columns_rejected():
    with pytest.raises(ValidationError, match="duplicate column entries"):
        TransformPlan(
            columns=[
                {"column": "a", "role": "X", "encoding": {"preset": "passthrough"}},
                {"column": "a", "role": "W", "encoding": {"preset": "drop"}},
            ]
        )


def test_transform_plan_requires_X_and_W():
    with pytest.raises(ValidationError, match="must contain at least one X"):
        TransformPlan(columns=[{"column": "w", "role": "W", "encoding": {"preset": "drop"}}])

    with pytest.raises(ValidationError, match="must contain at least one W"):
        TransformPlan(columns=[{"column": "x", "role": "X", "encoding": {"preset": "drop"}}])


# =============================================================================
# cat_onehot
# =============================================================================
def test_cat_onehot_impute_token_ok():
    enc = parse_enc(
        {
            "preset": "cat_onehot",
            "missing": "impute_token",
            "missing_token": "__MISSING__",
        }
    )
    assert enc.preset == "cat_onehot"
    assert enc.missing == "impute_token"
    assert enc.missing_token == "__MISSING__"


def test_cat_onehot_dummy_na_ok_and_token_ignored():
    enc = parse_enc(
        {
            "preset": "cat_onehot",
            "missing": "dummy_na",
            "missing_token": "__IGNORED__",
        }
    )
    assert enc.preset == "cat_onehot"
    assert enc.missing == "dummy_na"
    # token is present but validator intentionally doesn't care


def test_cat_onehot_error_ok_and_token_ignored():
    enc = parse_enc(
        {
            "preset": "cat_onehot",
            "missing": "error",
            "missing_token": "__IGNORED__",
        }
    )
    assert enc.preset == "cat_onehot"
    assert enc.missing == "error"


def test_cat_onehot_missing_token_cannot_be_empty():
    # NonEmptyStr should catch this (and also model_config strips whitespace)
    with pytest.raises(ValidationError):
        parse_enc(
            {
                "preset": "cat_onehot",
                "missing": "impute_token",
                "missing_token": "   ",
            }
        )


# =============================================================================
# num_minmax
# =============================================================================
def test_num_minmax_eps_must_be_positive():
    with pytest.raises(ValidationError, match="eps must be > 0"):
        parse_enc({"preset": "num_minmax", "eps": 0.0})

    with pytest.raises(ValidationError, match="eps must be > 0"):
        parse_enc({"preset": "num_minmax", "eps": -1.0})

    ok = parse_enc({"preset": "num_minmax", "eps": 1e-9})
    assert ok.preset == "num_minmax"
    assert ok.eps == 1e-9


# =============================================================================
# map_binary
# =============================================================================
def test_map_binary_requires_unknown_value_when_allow_unknown_true():
    with pytest.raises(ValidationError, match="unknown_value required when allow_unknown=True"):
        parse_enc(
            {
                "preset": "map_binary",
                "mapping": {"a": 1.0},
                "allow_unknown": True,
                "unknown_value": None,
                "missing": "error",
            }
        )


def test_map_binary_requires_unknown_value_when_missing_as_unknown():
    with pytest.raises(ValidationError, match="unknown_value required when missing='as_unknown'"):
        parse_enc(
            {
                "preset": "map_binary",
                "mapping": {"a": 1.0},
                "allow_unknown": False,
                "unknown_value": None,
                "missing": "as_unknown",
            }
        )


def test_map_binary_missing_impute_token_requires_missing_token():
    with pytest.raises(ValidationError, match="missing_token required when missing='impute_token'"):
        parse_enc(
            {
                "preset": "map_binary",
                "mapping": {"a": 1.0},
                "allow_unknown": True,
                "unknown_value": -1.0,
                "missing": "impute_token",
                "missing_token": None,
            }
        )


def test_map_binary_ok_happy_path():
    enc = parse_enc(
        {
            "preset": "map_binary",
            "mapping": {"yes": 1.0, "no": 0.0, "__NA__": -1.0},
            "allow_unknown": True,
            "unknown_value": -9.0,
            "missing": "impute_token",
            "missing_token": "__NA__",
        }
    )
    assert enc.preset == "map_binary"
    assert enc.mapping["yes"] == 1.0


# =============================================================================
# map_ordinal
# =============================================================================
def test_map_ordinal_rejects_duplicates():
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        parse_enc(
            {
                "preset": "map_ordinal",
                "order": ["a", "a"],
                "allow_unknown": True,
                "unknown_value": -1,
                "missing": "error",
            }
        )


def test_map_ordinal_missing_impute_token_requires_missing_token_and_position():
    with pytest.raises(ValidationError, match="missing_token and token_position required"):
        parse_enc(
            {
                "preset": "map_ordinal",
                "order": ["a", "b"],
                "allow_unknown": True,
                "unknown_value": -1,
                "missing": "impute_token",
                "missing_token": "__NA__",
                "token_position": None,
            }
        )


def test_map_ordinal_requires_unknown_value_when_allow_unknown_true():
    with pytest.raises(ValidationError, match="unknown_value required when allow_unknown=True"):
        parse_enc(
            {
                "preset": "map_ordinal",
                "order": ["a", "b"],
                "allow_unknown": True,
                "unknown_value": None,
                "missing": "error",
            }
        )


def test_map_ordinal_requires_unknown_value_when_missing_as_unknown():
    with pytest.raises(ValidationError, match="unknown_value required when missing='as_unknown'"):
        parse_enc(
            {
                "preset": "map_ordinal",
                "order": ["a", "b"],
                "allow_unknown": False,
                "unknown_value": None,
                "missing": "as_unknown",
            }
        )


def test_map_ordinal_ok_happy_path():
    enc = parse_enc(
        {
            "preset": "map_ordinal",
            "order": ["low", "mid", "high"],
            "start": 0,
            "allow_unknown": True,
            "unknown_value": -1,
            "missing": "impute_token",
            "missing_token": "__NA__",
            "token_position": "prepend",
        }
    )
    assert enc.preset == "map_ordinal"
    assert enc.order == ["low", "mid", "high"]


# =============================================================================
# discriminator + extra="forbid" sanity
# =============================================================================
def test_discriminator_rejects_unknown_preset():
    with pytest.raises(ValidationError):
        parse_enc({"preset": "does_not_exist"})


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        parse_enc({"preset": "num_standard", "some_extra": 123})
        

def test_encoding_preset_requires_discriminator_field():
    with pytest.raises(ValidationError):
        ENC.validate_python({"missing": "impute_token"})


def test_encoding_preset_coerces_numeric_strings_to_float():
    enc = ENC.validate_python(
        {
            "preset": "map_binary",
            "mapping": {"yes": "1.0", "no": 0.0},
            "allow_unknown": True,
            "unknown_value": -1.0,
            "missing": "error",
        }
    )
    assert enc.mapping["yes"] == 1.0


def test_encoding_preset_rejects_non_numeric_mapping_values():
    with pytest.raises(ValidationError):
        ENC.validate_python(
            {
                "preset": "map_binary",
                "mapping": {"yes": "not-a-number"},
                "allow_unknown": True,
                "unknown_value": -1.0,
                "missing": "error",
            }
        )