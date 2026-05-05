from pathlib import Path
import sys
import pandas as pd
from neo4j import GraphDatabase

# ---------------------------------------------------
# Make project root importable
# ---------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

# ---------------------------------------------------
# Neo4j connection
# ---------------------------------------------------
driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
)

# ---------------------------------------------------
# IMPROVED QUERY: Focus on 2020 and extract context
# ---------------------------------------------------
# We pull the entity, the specific case name, and the 
# holding to create a "Ground Truth" for evaluation.
QUERY = """
MATCH (c:Case)-[:APPLIES|INVOLVES]-(e)
WHERE (e:Statute OR e:LegalConcept) 
  AND c.date CONTAINS '2020'
  AND c.holdings IS NOT NULL 
  AND size(c.holdings) > 20

WITH e, c
ORDER BY rand()
LIMIT 40

RETURN
    labels(e)[0] as entity_type,
    e.name as entity_name,
    c.case_name as ground_truth_case,
    c.holdings as context_holding,
    c.case_number as case_id
"""

# ---------------------------------------------------
# Execute and Process
# ---------------------------------------------------
with driver.session() as session:
    rows = session.run(QUERY).data()

# Create DataFrame
df = pd.DataFrame(rows)

# Clean up holding text (remove newlines/extra spaces for CSV safety)
if not df.empty:
    df['context_holding'] = df['context_holding'].str.replace(r'\s+', ' ', regex=True).str.strip()

# ---------------------------------------------------
# Save results
# ---------------------------------------------------
output_dir = PROJECT_ROOT / "evaluation"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "eval_entities_2020.csv"

df.to_csv(output_file, index=False)

# ---------------------------------------------------
# Display
# ---------------------------------------------------
print(f"\n Generated {len(df)} 2020 evaluation anchors.")
print(f" Saved to: {output_file}\n")
print(df[['entity_name', 'ground_truth_case']].head(10).to_string(index=False))

driver.close()