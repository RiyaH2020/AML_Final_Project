from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from config import config
from neo4j import GraphDatabase


driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(
        config.NEO4J_USERNAME,
        config.NEO4J_PASSWORD
    )
)


queries = {

    "Judges": """
    MATCH (j:Judge)
    RETURN j.name as name
    LIMIT 20
    """,

    "Statutes": """
    MATCH (s:Statute)
    RETURN s.name as name
    LIMIT 20
    """,

    "Concepts": """
    MATCH (c:LegalConcept)
    RETURN c.name as name
    LIMIT 30
    """
}


with driver.session() as session:

    for label, query in queries.items():

        print()
        print("=" * 50)
        print(label.upper())
        print("=" * 50)

        rows = session.run(query).data()

        for row in rows:

            print(row["name"])


driver.close()