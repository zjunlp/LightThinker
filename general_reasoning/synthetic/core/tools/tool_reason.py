from typing import Dict


TOOLS_UNIFIED_FINAL = [
    {
        "type": "function",
        "function": {
            "name": "commit",
            "description": "Saves a reasoning step to memory. You must provide the raw 'detailed_reasoning' (archived for future extraction) and a 'summary' (kept visible in the context window).",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_reasoning": {
                        "type": "string",
                        "description": "The full, self-contained work for THIS STEP ONLY. Show all math, formulas, and logic."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A concise, one-sentence conclusion of the step. Example: 'The radius r was found to be s/√3.'"
                    }
                },
                "required": ["detailed_reasoning", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Retrieves the hidden 'detailed_reasoning' of a past step. Use this to EXTRACT specific formulas, values, or logic needed for your current reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "integer",
                        "description": "The ID number from the '[Step ID: N]' marker you want to expand."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explain explicitly what specific data (value/formula) you are extracting from this step."
                    }
                },
                "required": ["step_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fold",
            "description": "Collapses an expanded step back to summary view to free up context space. Mandatory action immediately after the extracted information has been utilized in a new 'commit'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "integer",
                        "description": "The ID number from an '[Expanded Step ID: N]' marker you want to collapse."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Confirm that the detailed information has been integrated into your latest reasoning and is no longer needed."
                    }
                },
                "required": ["step_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submits the final solution once the problem is completely solved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {
                        "type": "string",
                        "description": "The full final solution ending with the answer in a \\\\boxed{}. If the problem is a multiple-choice question and the answer is an option (e.g., A-E), then put the option letter inside \\\\boxed{} (e.g., \\\\boxed{A}). If there is no such option requirement, put the corresponding numerical or algebraic value inside \\\\boxed{} (e.g., \\\\boxed{-\\frac{2}{3}})."
                    }
                },
                "required": ["final_answer"]
            }
        }
    }
]


TOOLS_UNIFIED_FINAL_1229 = [
    {
        "type": "function",
        "function": {
            "name": "commit",
            "description": "Saves a reasoning step to memory. You must provide the raw 'detailed_reasoning' (archived for future extraction) and a 'summary' (kept visible in the context window).",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_reasoning": {
                        "type": "string",
                        "description": "The full, self-contained work for THIS STEP ONLY. Show all math, formulas, and logic."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A concise, one-sentence conclusion of the step. Example: 'The radius r was found to be s/√3.'"
                    }
                },
                "required": ["detailed_reasoning", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "expand",
            "description": "Retrieves the hidden 'detailed_reasoning' of a past step. Use this to EXTRACT specific formulas, values, or logic needed for your current reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "integer",
                        "description": "The ID number from the '[Step ID: N]' marker you want to expand."
                    },
                    "reason": {
                        "type": "string",
                        "description": "A concrete, task-oriented justification for expansion (e.g., 'extract the exact loss function definition', 'verify intermediate results', 'retrieve numerical constants used in calculation')."
                    }
                },
                "required": ["step_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fold",
            "description": "Collapses an expanded step back to summary view to free up context space. Mandatory action immediately after the extracted information has been utilized in a new 'commit'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_id": {
                        "type": "integer",
                        "description": "The ID number from an '[Expanded Step ID: N]' marker you want to collapse."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Confirmation that the extracted details have been successfully incorporated (e.g., 'formula recorded and reused', 'values verified and applied', 'logic integrated into current decision')."
                    }
                },
                "required": ["step_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submits the final solution once the problem is completely solved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer": {
                        "type": "string",
                        "description": "The full final solution ending with the answer in a \\\\boxed{}. If the problem is a multiple-choice question and the answer is an option (e.g., A-E), then put the option letter inside \\\\boxed{} (e.g., \\\\boxed{A}). If there is no such option requirement, put the corresponding numerical or algebraic value inside \\\\boxed{} (e.g., \\\\boxed{-\\frac{2}{3}})."
                    }
                },
                "required": ["final_answer"]
            }
        }
    }
]
