def get_compile_inference_node_info() -> str:
        return (
            "CompileInferenceNode: prepares an inference-ready dataset by dropping to protocol-required columns, "
            "purging missing values, applying exclusions, enforcing treatment/outcome domains, and persisting a cleaned "
            "dataset artifact. Returns INFERENCE_READY state with clean_dataset_id."
        )
