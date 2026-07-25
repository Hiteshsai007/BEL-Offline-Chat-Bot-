import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.retriever import get_retriever
from app.rag.pipeline import DIRECT_ANSWER_THRESHOLD

retriever = get_retriever()
queries = [
    "How does it work?",
    "How do I fix it?",
    "How does it work? 0x0003",
    "How do I fix it? 0x0003",
]

for q in queries:
    results = retriever.retrieve(q)
    print(f"\n==========================================")
    print(f"QUERY: '{q}'")
    print(f"==========================================")
    if not results:
        print("  No chunks retrieved.")
        continue
    
    top_score = results[0].score
    print(f"Top Score: {top_score:.4f} (Threshold = {DIRECT_ANSWER_THRESHOLD})")
    print(f"Fast Path Selected? -> {top_score >= DIRECT_ANSWER_THRESHOLD}")
    print("\nRetrieval Results:")
    for i, r in enumerate(results, 1):
        code = r.chunk.get("error_code") or "N/A"
        desc = r.chunk.get("error_description") or "N/A"
        doc = r.chunk.get("document_name") or "N/A"
        rem = r.chunk.get("error_remarks") or "N/A"
        print(f"  [{i}] Score: {r.score:.4f} | Code: {code} | Doc: {doc}")
        print(f"      Desc: {desc}")
        print(f"      Remarks: {rem}")
