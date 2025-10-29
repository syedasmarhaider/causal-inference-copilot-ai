from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Tuple, Any

TreatmentType = Literal["binary", "multi", "continuous"]
Identification = Literal["backdoor", "iv", "panel"]

@dataclass(frozen=True)
class ProblemSpec:
    identification: Identification
    treatment_type: TreatmentType
    outcome_type: Literal["continuous", "binary"]
    has_instruments: bool = False

class CateEstimator(ABC):
    """Minimal CATE/ATE contract across libraries."""
    def __init__(self, spec: ProblemSpec, **config: Any): ...
    @abstractmethod
    def fit(self, X, T, y, W=None, Z=None) -> "CateEstimator": ...
    @abstractmethod
    def cate(self, X) -> Any: ...                    # individual effects
    @abstractmethod
    def cate_interval(self, X, alpha=0.05) -> Tuple[Any, Any]: ...
    @abstractmethod
    def ate(self, X=None) -> float: ...              # overall or conditional ATE
    @abstractmethod
    def ate_interval(self, X=None, alpha=0.05) -> Tuple[float, float]: ...

class PolicyLearner(ABC):
    @abstractmethod
    def fit(self, X, T, y, W=None) -> "PolicyLearner": ...
    @abstractmethod
    def recommend(self, X) -> Any: ...               # treatment recommendation
    @abstractmethod
    def policy_value(self, X, T, y) -> float: ...    # off-policy eval

class Interpreter(ABC):
    @abstractmethod
    def fit(self, cate_model: CateEstimator, X_ref) -> "Interpreter": ...
    @abstractmethod
    def to_rules(self, max_leaves=8) -> str: ...     # one-tree summary
