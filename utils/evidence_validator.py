#!/usr/bin/env python3
"""
Evidence Validator & Citation Enricher (MERGED WITH FACT CHECKER)
Post-processes pipeline JSON to validate claims, fix inaccuracies, and ensure scientific rigor.
Uses Gemini 3.0 Pro (or best available reasoning model) with Google Search.
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from google.genai import types
from dotenv import load_dotenv
try:
    from utils.llm_response_parser import extract_json_from_llm_response
except ImportError:
    # Fallback if running as standalone script from root or utils
    try:
        from llm_response_parser import extract_json_from_llm_response
    except ImportError:
        # Fallback implementation
        def extract_json_from_llm_response(text: str) -> dict:
            cleaned = (text or "").strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].lstrip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(cleaned[start:end+1])
                raise

# Constants
MAX_THINKING_TOKENS = 32768
MAX_OUTPUT_TOKENS = 65536

class EvidenceValidatorError(RuntimeError):
    """Raised when evidence validation fails."""
    pass

def load_json_file(json_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to load JSON: {e}")

def save_json_file(data: Dict[str, Any], output_path: Path) -> None:
    try:
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        print(f"[OK]Saved validated output to: {output_path}")
    except Exception as e:
        raise EvidenceValidatorError(f"Failed to save JSON: {e}")

def call_gemini_with_search(
    prompt: str,
    api_key: str,
    verbose: bool = False
) -> str:
    """
    Call Gemini 3.0 Pro (or fallback) with maximum thinking budget and Google Search.
    """
    from google import genai as google_genai

    client = google_genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
        top_p=0.90,
    )

    if verbose:
        print(f"    Calling Gemini (Reasoning/Search Enabled)...")

    max_retries = 3
    base_delay = 5.0

    # Model selection: Prioritize 3.0 Pro Preview as requested
    # Fallback to Flash Thinking if 3.0 is unavailable
    models_to_try = ["gemini-3.0-pro-preview", "gemini-2.0-flash-thinking-exp-01-21"]

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1 and verbose:
                    print(f"    Trying model: {model_name}")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

                if hasattr(response, 'text'):
                    output = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    output = ''.join(part.text for part in parts if hasattr(part, 'text'))
                else:
                    raise EvidenceValidatorError("No text in response")

                return output.strip()

            except Exception as e:
                # Check if it's a 404 (model not found) or quota issue
                error_str = str(e)
                if "404" in error_str or "not found" in error_str.lower():
                    if verbose:
                        print(f"    [Warn] Model {model_name} not found. Trying next...")
                    break # Break attempt loop to try next model

                delay = base_delay * (2 ** (attempt - 1))
                if attempt < max_retries:
                    if verbose:
                        print(f"    [Retry] Error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # If strict retries failed for this model, try next model
                    if verbose:
                        print(f"    [Warn] Failed with {model_name} after retries.")
                    break
    
    raise EvidenceValidatorError("All models failed.")

def create_rigorous_validation_prompt(
    main_protein: str,
    batch: List[Dict[str, Any]],
    batch_start: int,
    batch_end: int,
    total_count: int
) -> str:
    """
    Create a DEEP FORENSIC AUDIT validation prompt.
    This prompts for a structured, field-by-field audit report similar to the manual example.
    """
    
    batch_json = json.dumps(batch, indent=2, ensure_ascii=False)
    
    prompt = f"""DEEP FORENSIC EVIDENCE AUDIT & CORRECTION TASK

You are a SKEPTICAL SCIENTIFIC AUDITOR. Your job is to conduct a forensic analysis of the claimed protein interaction.
**DEFAULT ASSUMPTION: The input claims are hallucinated or inaccurate until you find IRREFUTABLE primary literature.**

MAIN PROTEIN: {main_protein}
PROCESSING: Interactor {batch_start+1} of {total_count} (Batch size: 1 for maximum depth)

INPUT DATA (JSON):
{batch_json}

AUDIT PROTOCOL (MANDATORY):

1. **ADVERSARIAL SEARCH**:
   - Search for EVIDENCE THAT REFUTES THE CLAIM.
   - Search: `"{main_protein}" "{batch[0].get('primary')}" interaction`
   - Search: `"{main_protein}" "{batch[0].get('primary')}" NO interaction`
   - Search: `"{main_protein}" "{batch[0].get('primary')}" mechanism contradiction`
   - **Example Trap**: If input says "activates via deubiquitination", search for "represses transcription" or "inhibits degradation" to check for opposite mechanisms.

2. **DEEP FORENSIC CHECKS (The "Fact Check Verdict")**:
   - **Interaction**: Do they interact physically or just functionally? (Co-localization ≠ Interaction).
   - **Mechanism**: Is the specific mechanism (DUB, kinase, etc.) correct? Or is it a "transcriptional repressor" masquerading as a "deubiquitinase"?
   - **Effect**: Does it stabilize or degrade? Activate or inhibit?
   - **Role**: Is it an Oncogene or Tumor Suppressor in this context? (Opposite roles are common hallucinations).

3. **MANDATORY AUDIT REPORT FIELDS**:
   - For EVERY function, you must generate a structured audit:
     - `audit_verdict`: "Verified", "Refuted", "Corrected", "Unproven".
     - `mechanism_check`: "Accurate", "Incorrect Mechanism", "Opposite Effect", "Wrong Protein".
     - `scientific_consensus`: A 2-3 sentence summary of what the field ACTUALLY believes (e.g., "While early papers suggested X, Sacco et al. (2014) confirmed Y.").
     - `source_conflict`: "None", or "Claim contradicts Sacco et al. (2014)".
     - `confidence_score`: 1-10 (9+ required for Verified).

4. **CORRECTION LOGIC**:
   - If `audit_verdict` is "Corrected":
     - **REWRITE EVERYTHING**: Function name, arrow, cellular process, effect description.
     - **PROVIDE PROOF**: Exact paper title and relevant quote.
   - If `audit_verdict` is "Refuted" (e.g., wrong protein, no interaction):
     - Mark `validity: DELETED`.

OUTPUT FORMAT:
Return a JSON object with the exact structure of the input 'interactors' list, but ENRICHED with the Audit Report fields:
- Add to each function:
  - `validity`: 'TRUE' | 'CORRECTED' | 'FALSE' | 'DELETED'
  - `audit_verdict`: (String)
  - `mechanism_check`: (String)
  - `scientific_consensus`: (String)
  - `source_conflict`: (String)
  - `search_queries_performed`: (List of strings)
  - `confidence_score`: (Integer)
  - Update `evidence` array with REAL citations.

{batch_json}
"""
    return prompt

def validate_and_enrich_evidence(
    json_data: Dict[str, Any],
    api_key: str,
    verbose: bool = False,
    batch_size: int = 1, # DEFAULT TO 1 for maximum attention
    step_logger=None
) -> Dict[str, Any]:
    """
    Validate evidence using rigorous adversarial checking.
    """
    if 'ctx_json' not in json_data:
        return json_data

    ctx_json = json_data['ctx_json']
    interactors = ctx_json.get('interactors', [])
    main_protein = ctx_json.get('main', 'UNKNOWN')

    if not interactors:
        return json_data

    print(f"\n{'='*80}")
    print(f"DEEP FORENSIC EVIDENCE AUDIT FOR {main_protein}")
    print(f"{'='*80}")

    validated_interactors = []

    # Process in batches (Default 1)
    for batch_start in range(0, len(interactors), batch_size):
        batch_end = min(batch_start + batch_size, len(interactors))
        batch = interactors[batch_start:batch_end]

        print(f"  Auditing batch {batch_start//batch_size + 1}: Interactors {batch_start+1}-{batch_end}")

        prompt = create_rigorous_validation_prompt(main_protein, batch, batch_start, batch_end, len(interactors))

        try:
            response_text = call_gemini_with_search(prompt, api_key, verbose)
            result_json = extract_json_from_llm_response(response_text)

            # The result might be a list of interactors or a dict with 'interactors' key
            if isinstance(result_json, list):
                batch_results = result_json
            elif isinstance(result_json, dict) and 'interactors' in result_json:
                batch_results = result_json['interactors']
            else:
                # Fallback: try to match input structure
                batch_results = batch
                print("    [Warn] Failed to parse validation response structure. Keeping original.")

            # Process results: Remove DELETED, Handle CORRECTED
            for val_int in batch_results:
                # Check if entire interactor is deleted
                if val_int.get('validity') == 'DELETED':
                    print(f"    [DELETE] Removing {val_int.get('primary')} (Reason: {val_int.get('validation_note')})")
                    continue

                # Filter functions
                valid_functions = []
                for func in val_int.get('functions', []):
                    validity = func.get('validity', 'TRUE')

                    if validity == 'DELETED' or validity == 'FALSE':
                         print(f"    [DROP] Function '{func.get('function')}' for {val_int.get('primary')} ({validity})")
                         continue

                    if validity == 'CORRECTED':
                        print(f"    [CORRECTED] {val_int.get('primary')}: {func.get('function')} -> {func.get('mechanism_check')}")
                        print(f"       Consensus: {func.get('scientific_consensus')[:100]}...")

                    if validity == 'TRUE':
                        # Check confidence score if available
                        score = func.get('confidence_score')
                        if score and int(score) < 8: # Strict threshold
                             print(f"    [DROP] Function '{func.get('function')}' (Low Confidence: {score})")
                             continue
                        # Log the successful audit
                        if func.get('audit_verdict'):
                             print(f"    [VERIFIED] {val_int.get('primary')}: {func.get('function')} ({func.get('audit_verdict')})")

                    valid_functions.append(func)

                val_int['functions'] = valid_functions

                if not valid_functions and not val_int.get('validity') == 'TRUE':
                     print(f"    [DELETE] Removing {val_int.get('primary')} (No valid functions)")
                     continue

                validated_interactors.append(val_int)

        except Exception as e:
            print(f"    [Error] Batch validation failed: {e}")
            validated_interactors.extend(batch) # Fallback to original

    ctx_json['interactors'] = validated_interactors

    # Update snapshot if exists
    if 'snapshot_json' in json_data:
        json_data['snapshot_json']['interactors'] = validated_interactors

    return json_data

def main():
    """Main entry point for evidence validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate and enrich evidence in pipeline JSON output"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to the pipeline JSON file to validate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (default: <input>_validated.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of interactors to process per batch (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("[ERROR]GOOGLE_API_KEY not found. Add it to your .env file.")
    
    # Load input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        sys.exit(f"[ERROR]Input file not found: {input_path}")
    
    print(f"\n{'='*80}")
    print("EVIDENCE VALIDATOR & CITATION ENRICHER")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    
    json_data = load_json_file(input_path)

    # Validate and enrich with timing
    start_time = time.time()
    try:
        validated_data = validate_and_enrich_evidence(
            json_data,
            api_key,
            verbose=args.verbose,
            batch_size=args.batch_size
        )
    except Exception as e:
        sys.exit(f"[ERROR]Validation failed: {e}")

    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_validated{input_path.suffix}"

    save_json_file(validated_data, output_path)

    print(f"\n{'='*80}")
    print("[OK]VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_min:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
