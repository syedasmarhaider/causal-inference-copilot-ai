# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional

# from python.domain.repo.data_repo import DataRepo
# from python.domain.repo.models_repo import ModelsRepo

# from python.workflows.tools.inference.causal_inference import CausalInference
# from python.workflows.tools.inference.econml.dml import DML_FQCN, EconMLDMLInference


# @dataclass
# class CausalInferenceFactory:
#     _by_fqcn: Dict[str, CausalInference]

#     @classmethod
#     def create_default(cls, *, data_repo: DataRepo, models_repo: ModelsRepo) -> "CausalInferenceFactory":
#         dml = EconMLDMLInference(data_repo=data_repo, models_repo=models_repo)
#         return cls(_by_fqcn={DML_FQCN: dml})

#     def supported_estimators(self) -> List[str]:
#         return sorted(self._by_fqcn.keys())

#     def resolve(self, estimator_fqcn: str) -> Optional[CausalInference]:
#         return self._by_fqcn.get(estimator_fqcn)

#     def require(self, estimator_fqcn: str) -> CausalInference:
#         inf = self.resolve(estimator_fqcn)
#         if inf is None:
#             raise ValueError(f"Unsupported estimator_fqcn: {estimator_fqcn}. Supported: {self.supported_estimators()}")
#         return inf
