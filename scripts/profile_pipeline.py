"""
scripts/profile_pipeline.py

Performance profiling script for BEL Offline AI Assistant.
Measures timing across all key RAG & session memory stages.
"""
import sys
import time
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.pipeline import query
from app.session import get_session_store


def profile():
    session_id = f"profile_llm_{uuid.uuid4().hex[:8]}"
    store = get_session_store()

    # Pre-populate history with 3 conversation turns (6 messages) to simulate an active session
    for i in range(1, 4):
        store.add_turn(session_id, f"User question {i}", f"Assistant answer {i}")

    print(f"=== Starting LLM Path Profiling Run for Session: {session_id} ===")

    # Force low confidence to trigger LLM fallback path
    # Query with semantic hit score ~0.51 (below DIRECT_ANSWER_THRESHOLD 0.60)
    q_llm = "What causes fire abort?"
    print(f"\n--- Benchmark (LLM Path) --- Query: '{q_llm}'")
    res_llm = query(q_llm, session_id=session_id)
    print(f"Answer: {res_llm.answer[:100]}...")

    # Clean up test session
    store.clear_session(session_id)
    print("\n=== Profiling Complete ===")


if __name__ == "__main__":
    profile()
