def get_clean_protocol_node_info() -> str:
        return (
            "CleanProtocolNode: prepares an inference-ready dataset by dropping to protocol-required columns, "
            "purging missing values, applying exclusions, enforcing treatment/outcome domains, and persisting a cleaned "
            "dataset artifact. Returns state with clean_dataset_id."
        )
   