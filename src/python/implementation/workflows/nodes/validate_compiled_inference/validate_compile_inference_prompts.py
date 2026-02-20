def validate_compiled_inference_get_info() -> str:
        return (
            "Validate compiled inference inputs (clean dataset + compiled protocol) prior to transform/encoding. "
            "Produces FAIL/WARN issues and a user-facing summary."
        )
        
def system_prompt_validate_compiled_inference() -> str:
    return (
        "you will present validation error about data set and why it failed in medical terms so that clinicians can understand it."
    )  