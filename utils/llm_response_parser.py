"""
Utilities for parsing JSON responses from LLM models.
Handles markdown code fences and extra commentary.
"""
import json
import re
from typing import Any, Union


def extract_json_from_llm_response(text: str) -> Union[dict, list]:
    """
    Extract and parse JSON from an LLM response, handling:
    - Markdown code fences (```json ... ```)
    - Extra commentary before/after the JSON
    - Whitespace and formatting variations
    - JSON Objects {} and Arrays []

    Args:
        text: Raw response text from LLM

    Returns:
        Parsed JSON as dict or list

    Raises:
        ValueError: If no valid JSON can be extracted
        json.JSONDecodeError: If JSON is malformed
    """
    cleaned = (text or "").strip()

    # 1. Try to find JSON within markdown code blocks first
    # Look for ```json ... ``` or just ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, cleaned, re.IGNORECASE)
    
    # Try parsing each code block found
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # 2. If no valid code blocks, try to parse the whole text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Fallback: Find the largest possible JSON structure (object or array)
    # We look for the first '{' or '[' and the last '}' or ']'
    
    first_brace = cleaned.find("{")
    first_bracket = cleaned.find("[")
    
    start_index = -1
    if first_brace != -1 and first_bracket != -1:
        start_index = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_index = first_brace
    elif first_bracket != -1:
        start_index = first_bracket
        
    if start_index != -1:
        # Determine expected end character based on start
        is_object = cleaned[start_index] == "{"
        
        last_brace = cleaned.rfind("}")
        last_bracket = cleaned.rfind("]")
        
        end_index = -1
        if is_object:
             end_index = last_brace
        else:
             end_index = last_bracket
             
        if end_index > start_index:
            candidate = cleaned[start_index : end_index + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # If we still can't parse, raise with helpful message
    raise ValueError(
        f"Failed to parse JSON from LLM response. "
        f"Response length: {len(text)} chars, "
        f"Preview: {text[:200]}..."
    )
