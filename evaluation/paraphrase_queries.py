from pathlib import Path
import sys
import pandas as pd
import requests
import time
import re

# ---------------------------------------------------
# Make project root importable
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

# ---------------------------------------------------
# Configurations
# ---------------------------------------------------
# Input should now be the context-specific queries we generated
input_file = PROJECT_ROOT / "evaluation" / "eval_queries_2020.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Updated System Prompt to protect the 'Ground Truth' details
SYSTEM_PROMPT = """
You are an Indian legal research assistant.
Rewrite the legal query to sound like it was written by a professional lawyer or legal researcher.

Rules:
1. PRESERVE meaning and all specific identifiers.
2. DO NOT change the Case Names, Statute Numbers (e.g., Section 302, Article 21), or the year 2020.
3. Use formal legal vocabulary (e.g., 'jurisprudence', 'interpretation', 'ratio decidendi').
4. Produce EXACTLY ONE query.
5. Do not use bullet points or role prefixes (e.g., 'Lawyer:').
6. Output ONLY the rewritten query text.
"""

def clean_query(query: str, original: str):
    if not query or str(query).lower() == "nan":
        return original

    query = query.strip()
    
    # Take only the first line to avoid LLM "chatter"
    query = query.split("\n")[0].strip()

    # Remove markdown/bullets
    query = re.sub(r"^[-•*\s\d\.]+", "", query)

    # Remove common LLM prefixes
    query = re.sub(
        r"^(Lawyer|Law student|Legal researcher|Rewritten Query|Query)\s*:\s*",
        "",
        query,
        flags=re.I
    )

    # Safety: ensure it didn't strip the case name entirely
    if len(query) < 10:
        return original

    return query.strip()

def paraphrase(original_query: str):
    prompt = f"{SYSTEM_PROMPT}\n\nOriginal query:\n{original_query}"

    payload = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,  # Lower temperature for less "creative" hallucination
            "top_k": 40
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        return clean_query(raw, original_query)
    except Exception as e:
        print(f"  Error calling Ollama: {e}")
        return original_query

# ---------------------------------------------------
# Execution
# ---------------------------------------------------
if not input_file.exists():
    print(f" Error: {input_file} not found. Run query generation first.")
    sys.exit(1)

df = pd.read_csv(input_file)
paraphrased = []

print(f" Paraphrasing {len(df)} queries using model: {config.LLM_MODEL}")

for i, row in df.iterrows():
    original_query = row["query"]
    print(f"[{i+1}/{len(df)}] Processing: {row['query_id']}")
    
    final_query = paraphrase(original_query)
    paraphrased.append(final_query)
    
    # Small delay for local LLM stability
    time.sleep(0.1)

df["paraphrased_query"] = paraphrased

# Save results
output_file = PROJECT_ROOT / "evaluation" / "eval_queries_final_2020.csv"
df.to_csv(output_file, index=False)

print(f"\n Finished! Final evaluation set saved to: {output_file}")
print("-" * 30)
print(df[["query_id", "paraphrased_query"]].head().to_string(index=False))