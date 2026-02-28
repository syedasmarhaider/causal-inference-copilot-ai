import pytest
from pydantic import ValidationError

def _CausalSpec():
    from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
    return CausalSpec


def test_causal_spec_rejects_same_y_and_t():
    CausalSpec = _CausalSpec()
    with pytest.raises(ValidationError):
        CausalSpec.model_validate(
            {
                "Y": {"kind": "continuous", "column": "a"},
                "T": {"kind": "binary", "column": "a", "treated_values": [1], "control_values": [0]},
                "W": [],
                "X": [],
                "Z": [],
            }
        )


def test_causal_spec_rejects_y_in_w():
    CausalSpec = _CausalSpec()
    with pytest.raises(ValidationError):
        CausalSpec.model_validate(
            {
                "Y": {"kind": "continuous", "column": "y"},
                "T": {"kind": "binary", "column": "t", "treated_values": [1], "control_values": [0]},
                "W": ["y"],
                "X": [],
                "Z": [],
            }
        )


def test_causal_spec_rejects_t_in_x():
    CausalSpec = _CausalSpec()
    with pytest.raises(ValidationError):
        CausalSpec.model_validate(
            {
                "Y": {"kind": "continuous", "column": "y"},
                "T": {"kind": "binary", "column": "t", "treated_values": [1], "control_values": [0]},
                "W": [],
                "X": ["t"],
                "Z": [],
            }
        )


def test_causal_spec_rejects_w_x_overlap():
    CausalSpec = _CausalSpec()
    with pytest.raises(ValidationError):
        CausalSpec.model_validate(
            {
                "Y": {"kind": "continuous", "column": "y"},
                "T": {"kind": "binary", "column": "t", "treated_values": [1], "control_values": [0]},
                "W": ["a"],
                "X": ["a"],
                "Z": [],
            }
        )


def test_causal_spec_accepts_valid_roles():
    CausalSpec = _CausalSpec()
    spec = CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1], "control_values": [0]},
            "W": ["w1", "w2"],
            "X": ["x1"],
            "Z": ["z1"],
        }
    )
    assert spec.Y.column == "y"
    assert spec.T.column == "t"