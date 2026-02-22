def validate_cleaned_protocol_get_info() -> str:
        return (
            "Validate cleaned protocol inputs (clean dataset + compiled protocol) prior to transform/encoding. "
            "Produces FAIL/WARN issues and a user-facing summary."
        )
        
def system_prompt_validate_cleaned_protocol() -> str:
    return (
        "you will present validation error about data set and why it failed in medical terms so that clinicians can understand it."
        "for warning just warn dont ask for users opinion and say that you are going to transform data. For failure explain the actions user or system needs to do."
        "tell colnames too"
    )  