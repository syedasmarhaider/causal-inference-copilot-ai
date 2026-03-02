def get_clean_protocol_node_info() -> str:
        return (
            "CleanProtocolNode: prepares an inference-ready dataset by dropping to protocol-required columns, "
            "purging missing values, applying exclusions, enforcing treatment/outcome domains, and persisting a cleaned "
            "dataset artifact. Returns state with clean_dataset_id."
        )
   
 
   

CLEANING_MESSAGE_TEMPLATE = f"""
You are at the cleaning stage of the causal ML copilot.
You cleaned the data and applied exlcusion rules.
And created new dataset. Explain clinincan what has been dropped and why
and ask for acceptance to proceed with the cleaned dataset.
"""
    