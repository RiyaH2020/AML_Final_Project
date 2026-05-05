from pathlib import Path
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
random.seed(42)

# Load the new 2020 anchors
input_file = PROJECT_ROOT / "evaluation" / "eval_entities_2020.csv"
df = pd.read_csv(input_file)

# ---------------------------------------------------
# NEW: Context-Aware Templates
# ---------------------------------------------------
# These templates use the holding to create a "fact-based" question
# that is much easier to evaluate for accuracy.
STATUTE_TEMPLATES = [
    "In the 2020 case of {case}, how was {x} applied regarding {holding}?",
    "What was the Supreme Court's interpretation of {x} in the context of {holding}?",
    "Discuss the application of {x} as seen in the 2020 judgment: {case}."
]

CONCEPT_TEMPLATES = [
    "Explain how the doctrine of {x} was discussed in the 2020 judgment: {case}.",
    "How did the Supreme Court apply the principle of {x} regarding {holding}?",
    "In {case}, what legal precedent was established for {x}?"
]

def make_query(row):
    t = row["entity_type"]
    x = row["entity_name"]
    case = row["ground_truth_case"]
    
    # Take a meaningful snippet of the holding (approx 15-20 words)
    full_holding = str(row["context_holding"])
    holding_words = full_holding.split()
    holding_snippet = " ".join(holding_words[:20]) + "..."

    if t == "Statute":
        return random.choice(STATUTE_TEMPLATES).format(x=x, case=case, holding=holding_snippet)

    elif t == "LegalConcept":
        return random.choice(CONCEPT_TEMPLATES).format(x=x, case=case, holding=holding_snippet)

    return f"What was the 2020 ruling in {case} regarding {x}?"

# ---------------------------------------------------
# Process and Save
# ---------------------------------------------------
df["query"] = df.apply(make_query, axis=1)

# Ensure query_id is at the start
if "query_id" in df.columns:
    df.drop(columns=["query_id"], inplace=True)

df.insert(0, "query_id", [f"Q{i+1:02d}" for i in range(len(df))])

output_file = PROJECT_ROOT / "evaluation" / "eval_queries_2020.csv"
df.to_csv(output_file, index=False)

print(f"✅ Generated {len(df)} context-specific queries.")
print(df[["query_id", "query"]].head(5).to_string(index=False))