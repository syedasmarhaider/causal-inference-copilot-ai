from __future__ import annotations


def get_model_train_node_info() -> str:
    return (
        "Node for fitting the confirmed causal model against the active cleaned dataset "
        "using the already-confirmed inference-ready causal specification. It does not "
        "rebuild transformation plans or renegotiate column order; it trains directly "
        "from the confirmed preprocessing contract and stores the fitted model id."
    )
