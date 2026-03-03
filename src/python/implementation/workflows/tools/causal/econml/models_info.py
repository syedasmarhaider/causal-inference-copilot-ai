from __future__ import annotations

from pathlib import Path


# =============================================================================
# Linear DML Causal Model Info
# =============================================================================

ROOT_FILE = Path(__file__).resolve().parent / "info_txt"

def get_linear_dml_causal_model_info() -> str:
    with open(ROOT_FILE / "linear_dml.txt", "r", encoding="utf-8") as f:
        return f.read()
    
def get_sparse_linear_dml_causal_model_info() -> str:
    with open(ROOT_FILE / "sparse_linear_dml.txt", "r", encoding="utf-8") as f:
        return f.read()

def get_kernel_dml_causal_model_info() -> str:
    with open(ROOT_FILE / "kernel_dml.txt", "r", encoding="utf-8") as f:
        return f.read()

def get_causal_forest_dml_causal_model_info() -> str:
    with open(ROOT_FILE / "causal_forest_dml.txt", "r", encoding="utf-8") as f:
        return f.read()    

def get_linear_dr_learner_causal_model_info() -> str:
    with open(ROOT_FILE / "linear_dr.txt", "r", encoding="utf-8") as f:
        return f.read()            

def get_sparse_linear_dr_learner_causal_model_info() -> str:
    with open(ROOT_FILE / "sparse_linear_dr.txt", "r", encoding="utf-8") as f:
        return f.read()    

def get_forest_dr_learner_causal_model_info() -> str:
    with open(ROOT_FILE / "causal_forest_dr.txt", "r", encoding="utf-8") as f:
        return f.read()    


     
     
     

    