import logging
import re
import json5
from typing import Any, Dict, Tuple

def parse_model_output(text: str) -> Tuple[str, Dict[str, Any], bool]:
    """
    Parses the model output text to extract structured information.
    
    The function looks for specific XML-like tags in the text:
    - "... \n\n<commit> ... </commit>"
    - "... \n\n<expand> Step N </expand>"
    - "... \n\n<fold> Step N </fold>"
    - "\n\n<final_answer> ... </final_answer>"

    Depending on the tag found, it extracts and parses the outer and inner content.

    Args:
        text (str): The model output text.
    Returns:
        Tuple[str, dict, boolean]: A tuple containing the type of content extracted 
                         ("commit", "expand", "fold", "final_answer") and the parsed content as a dictionary. 
                         ("detailed_reasoning": outer raw content, "summary": inner raw content) for "commit", 
                         ("step_id": N) for "expand" and "fold", 
                         and {"final_answer": raw inner content} for "answer".    
    """ 
    patterns = {
        "final_answer": r"<answer>\s*(.*?)\s*(?:</answer>|$)",
        "expand": r"<expand>\s*(.*?)\s*(?:</expand>|$)",
        "fold": r"<fold>\s*(.*?)\s*(?:</fold>|$)",
        "commit": r"<commit>\s*(.*?)\s*(?:</commit>|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_inner_content = match.group(1).strip()
            
            # Case 1: Commit
            if key == "commit":
                detailed_reasoning = text[:match.start()].strip()
                
                if not detailed_reasoning and not raw_inner_content:
                    logging.warning(f"Found {key} tag but content is empty.")
                    return key, {}, False

                logging.info(f"Extracted {key}: reasoning_len={len(detailed_reasoning)}, summary_len={len(raw_inner_content)}")
                return key, {
                    "detailed_reasoning": detailed_reasoning, 
                    "summary": raw_inner_content
                }, True
            
            # Case 2: Expand / Fold
            elif key in ["expand", "fold"]:
                try:
                    id_match = re.search(r"(?:Step\s*)?(\d+)", raw_inner_content, re.IGNORECASE)
                    if id_match:
                        step_id = int(id_match.group(1))
                        logging.info(f"Extracted {key} id: {step_id}")
                        return key, {"step_id": step_id}, True
                    else:
                        logging.warning(f"Failed to extract Step ID from {key}. Content: '{raw_inner_content}'")
                        return key, {}, False
                except Exception as e:
                    logging.error(f"Error parsing ID for {key}: {e}", exc_info=True)
                    return key, {}, True
            
            # Case 3: Final Answer
            else: 
                logging.info(f"Extracted {key} content: ... {raw_inner_content[-50:]}")
                return key, {"final_answer": raw_inner_content}, True

    key = "commit"
    logging.warning(f"No XML tags found in model output. Defaulting to '{key}' with all text as reasoning.")
    return key, {
        "detailed_reasoning": text.strip(), 
        "summary": ""
    }, True



def parse_model_json_output(text: str) -> Tuple[str, Any]:
    patterns = {
        "commit": r"<commit>\s*(.*?)\s*(?:</commit>|$)",
        "expand": r"<expand>\s*(.*?)\s*(?:</expand>|$)",
        "fold": r"<fold>\s*(.*?)\s*(?:</fold>|$)",
        "final_answer": r"<final_answer>\s*(.*?)\s*(?:</final_answer>|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_inner_content = match.group(1).strip()

            if key == "commit":
                try: 
                    content = json5.loads(raw_inner_content) if raw_inner_content else None
                except ValueError:
                    content = extract_reasoning_and_summary(raw_inner_content)

                if content is None:
                    logging.info(f"Text: {text}")
                    logging.warning(f"Failed to extract detailed_reasoning for {key} . Returning empty.")
                    return key, {}
                
                return key, content          
            elif key in ["expand", "fold"]:
                try:
                    content = json5.loads(raw_inner_content)
                    logging.info(f"Content extracted for {key}: {content}...")
                except ValueError as e:
                    logging.error(f"Failed to parse JSON for {key}: {e}", exc_info=True)
                    return key, {}
                return key, content
            else:  # final_answer
                try: 
                    # First, try the standard strict JSON parser
                    parsed = json5.loads(raw_inner_content)
                    logging.info(f"Content extracted for {key}: {parsed}...")
                except ValueError:
                    logging.warning(f"Content for {key} is not valid JSON. Returning final answer as raw text.")
                    return key, {"raw_content": raw_inner_content}
                    
                    # # --- USE HELPER FUNCTION HERE ---
                    # extracted_text = extract_json_content(raw_inner_content, key)
                    
                    # # Re-construct valid dict structure
                    # parsed = {key: extracted_text}
                    # logging.info(f"Fallback extraction successful for {key}.")
                    
                return key, parsed

    # Default fallback (if no XML tags found)
    key = "commit"
    logging.warning(f"No XML tags found in model output. Defaulting to '{key}' with raw content extraction.")
    content = extract_reasoning_and_summary(text.strip())
    if content is None:
        return key, {}
    return key, content


def extract_reasoning_and_summary(content: str):
    # match "detailed_reasoning": ... , "summary": ...
    pattern = r'"detailed_reasoning":\s*(.*?)\s*,\s*"summary":\s*(.*?)\s*}'
    match = re.search(pattern, content, re.S)
    if match:
        detailed_reasoning = match.group(1).strip()
        summary = match.group(2).strip()
        return {
            "detailed_reasoning": detailed_reasoning,
            "summary": summary
        }
    else:
        return None

def extract_json_content(text: str, key: str) -> str:
    """
    Attempts to extract content from a string that looks like JSON 
    but might be malformed (e.g., unquoted text).
    
    Pattern logic:
    1. Starts with optional whitespace and {
    2. Matches "key":
    3. Captures everything (.) until the very end, 
       handling the case where the closing bracket might be at the end.
    """
    # Regex to match: {"key": <content> } (ignoring strict JSON rules)
    # We capture the rest of the string after the key
    pattern = r'^\s*\{\s*"' + re.escape(key) + r'"\s*:\s*(.*)'
    
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # If the model added a closing '}', remove it.
        # We check if it ends with '}' allowing for trailing newlines/spaces
        if content.endswith('}'):
            content = content[:-1].strip()
        return content
    
    # If the pattern didn't match (maybe the model didn't output {"key": at all),
    # just return the raw text as a fallback.
    return text.strip()