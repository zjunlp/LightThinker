

expandboth_deepresearch_tool = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Expands a summarized Thought or Observation in your memory log to view its full, original content. Use this when a summary lacks the specific detail needed for your next step.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought_id": {
                    "type": "integer",
                    "description": "The ID number from a `[Thought ID: N]` marker to see your full, original thought process."
                },
                "obs_id": {
                    "type": "integer",
                    "description": "The ID number from an `[Observation ID: N]` marker to see the full, raw data from that observation."
                },
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you are expanding this item and what specific details you are looking for."
                }
            },
            # "properties_optional": ["thought_id", "obs_id"]
        }
    }
}

foldboth_deepresearch_tool = {
    "type": "function",
    "function": {
        "name": "fold",
        "description": "Collapses an expanded Thought or Observation back to its summary view. Use this to keep your memory log clean and focused after you have extracted the necessary details.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought_id": {
                    "type": "integer",
                    "description": "The ID number from a `[Expanded Thought ID: N]` marker to collapse it back to its summary view."
                },
                "obs_id": {
                    "type": "integer",
                    "description": "The ID number from a `[Expanded Observation ID: N]` marker to collapse it back to its summary view."
                },
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you are folding this item back to its summary view."
                }
            },
        }
    }
}

expand_deepresearch_tool = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Expands a summarized Observation in your memory log to view its full, original content. Use this when a summary lacks the specific detail needed for your next step.",
        "parameters": {
            "type": "object",
            "properties": {
                "obs_id": {
                    "type": "integer",
                    "description": "The ID number from an `[Observation ID: N]` marker to see the full, raw data from that observation."
                },
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you are expanding this item and what specific details you are looking for."
                }
            },
        }
    }
}

final_answer_tool_v6 = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Call this function to submit the final solution once the problem is completely solved.",
        "parameters": {
            "type": "object",
            "properties": {
                "final_answer": {
                    "type": "string",
                    "description": (
                        "The complete and final solution text. "
                        "It MUST include the final numerical answer enclosed in \\\\boxed{}. "
                        "For example: 'The final area is \\\\boxed{78.5}'."
                    )
                }
            },
            "required": ["final_answer"]
        }
    }
}


fold_deepresearch_tool = {
    "type": "function",
    "function": {
        "name": "fold",
        "description": "Collapses an expanded Observation back to its summary view. Use this to keep your memory log clean and focused after you have extracted the necessary details.",
        "parameters": {
            "type": "object",
            "properties": {
                "obs_id": {
                    "type": "integer",
                    "description": "The ID number from an `[Expanded Observation ID: N]` marker to collapse it back to its summary view."
                },
                "reason": {
                    "type": "string",
                    "description": "A brief explanation of why you are folding this item back to its summary view."
                }
            },
        }
    }
}



log_step_toolv0 = {
    "type": "function",
    "function": {
        "name": "log_step",
        "description": (
            "Analyzes the full record of your last action (both your thought process and the resulting observation) and archives summarized versions into your permanent memory. This is the main tool for saving your progress."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought_summary": {
                    "type": "string",
                    "description": "A concise, 1-sentence summary of your original thought/plan for the action."
                },
                "obs_summary": {
                    "type": "string",
                    "description": (
                        "A concise summary of the key findings from the raw observation. **CRITICAL**: This summary must act as a 'future index'. "
                        "Include key findings AND 'hooks' (names, numbers, or hints of other available info) so you know which steps are worth expanding later.\n\n"
                        "**Good Example**: 'Found several frameworks: LangChain, AutoGen, and CrewAI. The search also returned a paper on the ReAct technique.'\n"
                        "**Bad Example**: 'The search was successful.'"
                    )
                },
                "detailed_analysis": {
                    "type": "string",
                    "description": "Your detailed analysis of how the observation fulfills (or doesn't fulfill) your original thought. This justifies your summaries."
                }
            },
            "required": ["thought_summary", "obs_summary", "detailed_analysis"]
        }
    }
}

log_step_toolv1 = {
    "type": "function",
    "function": {
        "name": "log_step",
        "description": (
            "Analyzes the 'Temporary Action Record' to log progress and strategize the next move. "
            "This is your primary tool for thinking like a detective, building a case step-by-step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought_summary": {
                    "type": "string",
                    "description": "A 1-sentence summary of your specific plan for the single action you just took (your `Raw Thought`). Do not summarize the entire mission."
                },
                "obs_summary": {
                    "type": "string",
                    "description": (
                        "A strategic 'Table of Contents' of the `Raw Observation`. Its purpose is to create an index for future investigation. "
                        "**CRITICAL RULE**: This summary MUST list discoverable resources (e.g., 'Wikipedia page', 'IMDb filmography', 'article titled X') but MUST NOT state the facts or conclusions contained within those resources. "
                        "Think of it as creating 'hooks' for your future self to use with the `expand` or `visit` tools.\n\n"
                        "**CORRECT EXAMPLE (Strategic Index)**: 'The search returned a Wikipedia page for Ginnifer Goodwin and her IMDb filmography, which may contain the 'Erica' role.'\n"
                        "**INCORRECT EXAMPLE (Greedy Conclusion)**: 'Ginnifer Goodwin played Erica in Law & Order.'"
                    )
                },
                "detailed_analysis": {
                    "type": "string",
                    "description": "Your brief analysis connecting your `Raw Thought` to the clues listed in your `obs_summary`. Explain why the resources you found are relevant to your next step."
                }
            },
            "required": ["thought_summary", "obs_summary", "detailed_analysis"]
        }
    }
}

log_step_tool = {
    "type": "function",
    "function": {
        "name": "log_step",
        "description": (
            "Analyzes the 'Temporary Action Record' to log progress and strategize the next move. "
            "This is your primary tool for thinking like a detective, building a case step-by-step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought_summary": {
                    "type": "string",
                    "description": "A 1-sentence summary of your specific plan for the single action you just took (your `Raw Thought`). Do not summarize the entire mission."
                },
                "obs_summary": {
                    "type": "string",
                    "description": (
                        "A strategic 'Table of Contents' of the `Raw Observation`. Its purpose is to create an index for future investigation. "
                        "**CRITICAL RULE**: This summary MUST list discoverable resources (e.g., 'Wikipedia page', 'IMDb filmography', 'article titled X') but MUST NOT state the facts or conclusions contained within those resources. "
                        "Think of it as creating 'hooks' for your future self to use with the `expand` or `visit` tools.\n\n"
                        "**CORRECT EXAMPLE (Strategic Index)**: 'The search returned a Wikipedia page for Ginnifer Goodwin and her IMDb filmography, which may contain the 'Erica' role.'\n"
                        "**INCORRECT EXAMPLE (Greedy Conclusion)**: 'Ginnifer Goodwin played Erica in Law & Order.'"
                    )
                },
                "detailed_analysis": {
                    "type": "string",
                    "description": "Your brief analysis connecting your `Raw Thought` to the clues listed in your `obs_summary`. Explain why the resources you found are relevant to your next step."
                }
            },
            "required": ["thought_summary", "obs_summary", "detailed_analysis"]
        }
    }
}