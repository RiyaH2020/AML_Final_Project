from pathlib import Path
import sys
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.vector.vector_store import LegalVectorStore


vs = LegalVectorStore()


print()
print("=" * 50)
print("TOTAL CHUNKS")
print("=" * 50)

print(vs.count())


print()
print("=" * 50)
print("UNIQUE CASES")
print("=" * 50)

cases = vs.list_cases()

print(len(cases))


print()
print("=" * 50)
print("TOP DUPLICATED CASES")
print("=" * 50)


all_meta = vs._col.get(include=["metadatas"])["metadatas"]

case_ids = [
    m.get("case_number", "")
    for m in all_meta
]

counts = Counter(case_ids)

for case_id, count in counts.most_common(20):

    print(count, case_id)