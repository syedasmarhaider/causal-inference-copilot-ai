from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

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
from python.implementation.workflows.tools.causal.inference.econml.utils import now_utc

if TYPE_CHECKING:
    from python.implementation.workflows.tools.causal.inference.econml.dml._base_run_dml import (
        _BaseRunDML,
    )


@dataclass(frozen=True, slots=True)
class _BaseDMLAdapter(CausalModel):
    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    ESTIMATOR_CLS: ClassVar[Any]
    BACKEND_NAME: ClassVar[str]
    INFO: ClassVar[str]
    FIT_INCLUDE_NAMES: ClassVar[set[str]]
    USE_PRE_X_AS_FEATURIZER: ClassVar[bool] = True
    REQUIRE_NUMERIC_X: ClassVar[bool] = False
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = False
    CATE_QUERY_AS_NUMPY: ClassVar[bool] = False

    def get_info(self) -> str:
        return self.INFO

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(self.ESTIMATOR_CLS.fit) or ""
                base_doc = inspect.getdoc(self.ESTIMATOR_CLS) or ""
                return base_doc + fit_doc
            case "ATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.ate) or ""
            case "CATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.effect) or ""
            case _:
                return None

    def _build_run_dml(self) -> _BaseRunDML:
        from python.implementation.workflows.tools.causal.inference.econml.dml._base_run_dml import (
            _BaseRunDML,
        )

        return _BaseRunDML(
            models_repo=self.models_repo,
            encoding_util=self.encoding_util,
            estimator_cls=self.ESTIMATOR_CLS,
            backend_name=self.BACKEND_NAME,
            fit_include_names=self.FIT_INCLUDE_NAMES,
            use_pre_x_as_featurizer=self.USE_PRE_X_AS_FEATURIZER,
            require_numeric_x=self.REQUIRE_NUMERIC_X,
            drop_first_effect_modifier_onehot=self.DROP_FIRST_EFFECT_MODIFIER_ONEHOT,
            cate_query_as_numpy=self.CATE_QUERY_AS_NUMPY,
        )

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult:
        run_dml = self._build_run_dml()
        if isinstance(command, ValidateCommand):
            from python.implementation.workflows.tools.causal.inference.econml.dml.validate_dml import (
                _BaseValidateDML,
            )

            return _BaseValidateDML(run_dml=run_dml).execute(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
            )

        if isinstance(command, FitCommand):
            return run_dml.fit(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=command.df,
                started_at=now_utc(),
            )
        if isinstance(command, ATECommand):
            return run_dml.ate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=command.df,
                started_at=now_utc(),
            )
        if isinstance(command, CATECommand):
            return run_dml.cate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                started_at=now_utc(),
            )
        raise ValueError(f"Unsupported DML command: {type(command)}")
