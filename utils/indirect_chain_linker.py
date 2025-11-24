#!/usr/bin/env python3
"""
Indirect Chain Linker
Analyzes indirect interactions in the pipeline output and generates specific
function descriptions for the Mediator -> Indirect interaction in the context
of the original query chain. Saves these interactions to the protein database.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from google import genai as google_genai
from google.genai import types
from dotenv import load_dotenv

import utils.protein_database as pdb

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
MAX_THINKING_TOKENS = 8192  # Budget for chain analysis
MAX_OUTPUT_TOKENS = 8192

def _coerce_token_count(value: Any) -> int:
    """Best-effort conversion of token counts to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0

def call_gemini_chain_link(
    prompt: str,
    api_key: str,
    verbose: bool = False
) -> str:
    """
    Call Gemini 3.0 Pro to analyze the chain and generate function details.
    """
    client = google_genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=MAX_THINKING_TOKENS,
            include_thoughts=True,
        ),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )

    if verbose:
        print(f"    Requesting chain analysis from Gemini 3.0 Pro...")

    # Model priority: 3.0 Pro Preview -> Flash Thinking
    models_to_try = ["gemini-2.5-pro"]

    for model_name in models_to_try:
        try:
            if verbose:
                print(f"    Trying model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                return ''.join(part.text for part in parts if hasattr(part, 'text')).strip()

        except Exception as e:
            if verbose:
                print(f"    [Warn] Call failed for {model_name}: {e}")
            # Continue to next model
            continue

    return ""

def process_indirect_chain(
    ctx_json: Dict[str, Any],
    api_key: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main entry point to process indirect chains.
    """
    load_dotenv()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("[IndirectLinker] No API key found. Skipping.")
        return ctx_json

    main_protein = ctx_json.get("main")
    interactors = ctx_json.get("interactors", [])

    # Identify indirect interactors
    indirect_interactors = [
        i for i in interactors
        if i.get("interaction_type") == "indirect" and i.get("upstream_interactor")
    ]

    if not indirect_interactors:
        if verbose:
            print("[IndirectLinker] No indirect interactors found.")
        return ctx_json

    print(f"\n{'='*80}")
    print(f"INDIRECT CHAIN LINKER for {main_protein}")
    print(f"{'='*80}")
    print(f"Found {len(indirect_interactors)} indirect interactions to process.")

    linked_count = 0

    for interactor in indirect_interactors:
        target_name = interactor.get("primary")
        mediator_name = interactor.get("upstream_interactor")

        # Get the context from the indirect interactor's functions
        existing_functions = interactor.get("functions", [])
        context_description = ""
        if existing_functions:
            # Use the first function as context
            f = existing_functions[0]
            context_description = f"Function: {f.get('function')}\nProcess: {f.get('cellular_process')}\nChain: {main_protein} -> {mediator_name} -> {target_name}"
        else:
            context_description = f"Chain: {main_protein} -> {mediator_name} -> {target_name}"

        prompt = f"""
You are analyzing a protein interaction chain: {main_protein} -> {mediator_name} -> {target_name}.

The user is querying {main_protein}.
We found that {target_name} is an INDIRECT interactor, mediated by {mediator_name}.

Your task is to describe the SPECIFIC MOLECULAR MECHANISM by which {mediator_name} (Mediator) acts on {target_name} (Target), explicitly detailing how this enables the biological effect of the full chain ({main_protein} context).

Context of the chain:
{context_description}

RIGOROUS REQUIREMENTS:
1. **Mechanism**: Do not just say "regulates". Specify *how* (e.g., "recruits to promoter", "phosphorylates at Ser45", "ubiquitinates for degradation").
2. **Causality**: Explain why {mediator_name} is NECESSARY for {main_protein}'s effect on {target_name}.
3. **Direction**: Determine if {mediator_name} activates or inhibits {target_name}.
4. **Red Flags**: Avoid generic terms like "is associated with". We need physical or functional causality.

Please research and generate a structured function entry for the interaction:
{mediator_name} -> {target_name}

Return ONLY valid JSON in this format:
{{
  "function": "Specific Molecular Mechanism (e.g. 'Mediator Recruitment to Promoter')",
  "arrow": "activates|inhibits|binds",
  "cellular_process": "Detailed description of how {mediator_name} physically or functionally interacts with {target_name} to execute the chain's effect...",
  "effect_description": "One sentence summary of the net effect.",
  "biological_consequence": [
    "Step 1: {main_protein} activates {mediator_name}...",
    "Step 2: {mediator_name} binds/modifies {target_name}...",
    "Step 3: {target_name} executes downstream function..."
  ],
  "specific_effects": ["Effect 1", "Effect 2"],
  "evidence": [
    {{
      "paper_title": "Exact Title of a paper confirming {mediator_name}-{target_name} interaction",
      "year": 2024
    }}
  ]
}}
"""
        if verbose:
            print(f"  Processing chain: {main_protein} -> {mediator_name} -> {target_name}")

        result_text = call_gemini_chain_link(prompt, api_key, verbose=verbose)

        try:
            result_json = extract_json_from_llm_response(result_text)
        except Exception as e:
            if verbose:
                print(f"    [Warn] Failed to parse JSON for {target_name}: {e}")
            continue

        if result_json:
            # Construct the interaction object
            interaction_data = {
                "primary": target_name,
                "arrow": result_json.get("arrow"),
                "functions": [result_json],
                "interaction_type": "direct", # It is direct between mediator and target
                "discovered_in_query": main_protein # Track origin
            }

            # Load existing to append
            existing_interactions = pdb.get_all_interactions(mediator_name)
            # Find the specific one for target_name
            existing_int = next((i for i in existing_interactions if i.get("primary") == target_name), None)

            if existing_int:
                existing_funcs = existing_int.get("functions", [])
                existing_funcs.append(result_json)
                existing_int["functions"] = existing_funcs
                interaction_data = existing_int

            # Save Mediator -> Target interaction
            if verbose:
                print(f"    Saving linked interaction: {mediator_name} -> {target_name}")

            pdb.save_interaction(mediator_name, target_name, interaction_data)
            linked_count += 1

            interactor["_linked_mediator_interaction"] = f"{mediator_name}->{target_name}"

    print(f"Successfully linked {linked_count} indirect chains in database.")
    return ctx_json

if __name__ == "__main__":
    # Test run logic
    if len(sys.argv) > 1:
        # Allows running with a file input
        input_file = sys.argv[1]
        if os.path.exists(input_file):
            try:
                with open(input_file, 'r') as f:
                    data = json.load(f)
                ctx = data.get("ctx_json", data) # Handle both full payload and just ctx
                process_indirect_chain(ctx, "", verbose=True)
            except Exception as e:
                print(f"Error running test: {e}")
