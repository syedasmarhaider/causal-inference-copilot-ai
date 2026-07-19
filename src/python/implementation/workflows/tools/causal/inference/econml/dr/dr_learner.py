from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

from econml.dr import ForestDRLearner, LinearDRLearner, SparseLinearDRLearner

from python.domain.repo.models_repo import ModelsRepo
from python.implementation.workflows.tools.causal.encoding.encoding_util import EncodingUtil
from python.implementation.workflows.tools.causal.inference.causal_command import (
    ATECommand,
    CATECommand,
    CommandType,
    FitCommand,
    ValidateCommand,
)
from python.implementation.workflows.tools.causal.inference.causal_model import (
    CausalCommand,
    CausalModel,
    CausalResult,
)
from python.implementation.workflows.tools.causal.inference.econml.models_info import (
    get_forest_dr_learner_causal_model_info,
    get_linear_dr_learner_causal_model_info,
    get_sparse_linear_dr_learner_causal_model_info,
)
from python.implementation.workflows.tools.causal.inference.econml.utils import now_utc

if TYPE_CHECKING:
    from python.implementation.workflows.tools.causal.inference.econml.dr._base_run_dr import (
        _BaseRunDR,
    )


@dataclass(frozen=True, slots=True)
class _BaseDRLearnerAdapter(CausalModel):
    """Model-specific metadata plus command routing for EconML DR learners."""

    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    ESTIMATOR_CLS: ClassVar[Any]
    BACKEND_NAME: ClassVar[str]
    INFO: ClassVar[str]
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = False

    def get_info(self) -> str:
        return self.INFO

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                return (inspect.getdoc(self.ESTIMATOR_CLS) or "") + (
                    inspect.getdoc(self.ESTIMATOR_CLS.fit) or ""
                )
            case "ATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.ate) or ""
            case "CATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.effect) or ""
            case _:
                return None

    def _build_run_dr(self) -> _BaseRunDR:
        from python.implementation.workflows.tools.causal.inference.econml.dr._base_run_dr import (
            _BaseRunDR,
        )

        return _BaseRunDR(
            models_repo=self.models_repo,
            encoding_util=self.encoding_util,
            estimator_cls=self.ESTIMATOR_CLS,
            backend_name=self.BACKEND_NAME,
            drop_first_effect_modifier_onehot=self.DROP_FIRST_EFFECT_MODIFIER_ONEHOT,
        )

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult:
        run_dr = self._build_run_dr()
        if isinstance(command, ValidateCommand):
            from python.implementation.workflows.tools.causal.inference.econml.dr.validate_dr import (
                _BaseValidateDR,
            )

            return _BaseValidateDR(run_dr=run_dr).execute(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
            )
        if isinstance(command, FitCommand):
            return run_dr.fit(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=command.df,
                started_at=now_utc(),
            )
        if isinstance(command, ATECommand):
            return run_dr.ate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=command.df,
                started_at=now_utc(),
            )
        if isinstance(command, CATECommand):
            return run_dr.cate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                started_at=now_utc(),
            )
        raise ValueError(f"Unsupported DR command: {type(command)}")


@dataclass(frozen=True, slots=True)
class LinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = LinearDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.LinearDRLearner"
    INFO: ClassVar[str] = get_linear_dr_learner_causal_model_info()
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = True


@dataclass(frozen=True, slots=True)
class ForestDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = ForestDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.ForestDRLearner"
    INFO: ClassVar[str] = get_forest_dr_learner_causal_model_info()


@dataclass(frozen=True, slots=True)
class SparseLinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = SparseLinearDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.SparseLinearDRLearner"
    INFO: ClassVar[str] = get_sparse_linear_dr_learner_causal_model_info()
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = True
