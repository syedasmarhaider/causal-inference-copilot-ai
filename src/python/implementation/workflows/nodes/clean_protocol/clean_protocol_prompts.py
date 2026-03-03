def get_clean_protocol_node_info() -> str:
        return (
            "CleanProtocolNode: prepares an inference-ready dataset by dropping to protocol-required columns, "
            "purging missing values, applying exclusions, enforcing treatment/outcome domains, and persisting a cleaned "
            "dataset artifact. Returns state with clean_dataset_id."
            "Cleaning does not require user's input and is automated, but it requires user acceptance after cleaning to proceed with the cleaned dataset or not."
            ""
        )
   
 
   

CLEANING_MESSAGE_TEMPLATE = f"""
You are at the cleaning stage of the causal ML copilot.
You cleaned the data and applied exlcusion rules.
And created new dataset. Explain clinincan what has been dropped and why
and ask for acceptance to proceed with the cleaned dataset.
assing None whenever you are not sure about the acceptance and asking for confirmation from the user, True when you are sure that the user will accept and False when you are sure that the user have rejected.

"""
    