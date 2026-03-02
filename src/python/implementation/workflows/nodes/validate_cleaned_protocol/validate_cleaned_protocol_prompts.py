def validate_cleaned_protocol_get_info() -> str:
        return (
            "Validate cleaned protocol inputs (clean dataset + compiled protocol) prior to transform/encoding. "
            "Produces FAIL/WARN issues and a user-facing summary."
        )
        

VALIDATE_CLEAN_PROTOCOL_PROMPT = """
You are at the validaing clean protcol stage of the causal ML copilot for clinicians.
You task is to explain to the user how and why validation issues would effect the validity of the causal inference results and ask for user acceptance to proceed with the cleaned dataset and protocol or to go back to the protocol defining step.
If validation has hard fail then you should not ask for user acceptance and just explain the issues and say that user needs to go back to protocol defining step to fix them.
if validatin has warnings but no hard fails, explain the issues and ask for user acceptance to proceed or go back to protocol defining step to fix them.
No Data science jargon, explain in simple terms that clinicians can understand.

Output format:
JSON with 2 fields:
- "message_of_user": str,
- "user_acceptance": bool (true if user accepts to proceed with current cleaned dataset and protocol, false if user wants to go back to protocol defining step to fix the issues)

""".strip()
