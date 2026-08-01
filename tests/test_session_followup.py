"""
Tests for enhanced session-memory follow-up experience on fault codes and topic shift handling.
"""
import uuid
import pytest
from app.rag.pipeline import query, _resolve_history_context
from app.session import get_session_store


@pytest.fixture
def session_id():
    sid = str(uuid.uuid4())
    store = get_session_store()
    store.clear_session(sid)
    yield sid
    store.clear_session(sid)


def test_followup_how_do_i_fix_it(session_id):
    """Test Q1: Explain error 0x0003 -> Q2: How do I fix it?"""
    res1 = query("Explain error 0x0003", session_id=session_id)
    assert res1.found is True
    assert "0x0003" in res1.answer

    res2 = query("How do I fix it?", session_id=session_id)
    assert res2.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res2.answer
    assert "Available documented information:" in res2.answer
    assert "• Fire aborted" in res2.answer
    assert "[Source: IRL Fault Codes.pdf, 0x0003]" in res2.answer
    # Must not be duplicate metadata dump line
    assert res2.answer != res1.answer


def test_followup_what_is_the_remedy(session_id):
    """Test Q1: Explain error 0x0003 -> Q2: What is the remedy?"""
    query("Explain error 0x0003", session_id=session_id)
    res2 = query("What is the remedy?", session_id=session_id)
    assert res2.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res2.answer
    assert "• Fire aborted" in res2.answer


def test_followup_what_are_the_corrective_steps(session_id):
    """Test Q1: Explain error 0x0003 -> Q2: What are the corrective steps?"""
    query("Explain error 0x0003", session_id=session_id)
    res2 = query("What are the corrective steps?", session_id=session_id)
    assert res2.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res2.answer


def test_followup_what_action_should_i_take(session_id):
    """Test Q1: Explain error 0x0003 -> Q2: What action should I take?"""
    query("Explain error 0x0003", session_id=session_id)
    res2 = query("What action should I take?", session_id=session_id)
    assert res2.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res2.answer


def test_full_history_multi_turn_resolution(session_id):
    """
    Test full-history resolution across multiple turns:
    Q1: Explain error 0x0003
    Q2: Is it dangerous?
    Q3: What is the remedy?
    """
    res1 = query("Explain error 0x0003", session_id=session_id)
    assert res1.found is True

    # Intermediate turn without fault code in query
    res2 = query("Is it dangerous?", session_id=session_id)
    assert res2.found is True

    # Q3 asks for remedy — should still resolve 0x0003 from turn 1 history
    res3 = query("What is the remedy?", session_id=session_id)
    assert res3.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res3.answer


def test_topic_shift_from_fault_code_to_general_topic(session_id):
    """
    Test topic shift resolution:
    Q1: Explain error 0x0003  (exact code lookup — no embedding needed)
    Q2: How do I fix it?      (exact code lookup via context expansion — no embedding needed)
    Q3: How to start a bike?  (injected directly — no retrieval/embedding/Ollama dependency)
    Q4: _resolve_history_context verifies the topic shift is detected correctly.

    Expected:
    After Q3, context resolves to general_topic, NOT 0x0003.
    This validates session-memory topic switching, not retrieval quality.
    """
    res1 = query("Explain error 0x0003", session_id=session_id)
    assert res1.found is True

    res2 = query("How do I fix it?", session_id=session_id)
    assert res2.found is True
    assert "0x0003" in res2.answer

    # Inject the topic-shift turn directly.
    # _resolve_history_context only reads what is in history — it does not
    # care how the turn was created. This makes the test deterministic and
    # independent of retrieval scores, HF model availability, or Ollama.
    store = get_session_store()
    store.add_turn(
        session_id,
        "How to start a bike?",
        "To start a bike, first ensure the side stand is up.",
    )

    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What is the first step to do it?", history
    )

    # Must NOT resolve 0x0003 because topic shifted to bike starting in Q3.
    assert code is None
    assert source == "general_topic"
    assert exp == "What is the first step to do it?"


def test_resolve_history_context_unit():
    """Unit test for _resolve_history_context intent detection and resolution."""
    history = [
        {"role": "user", "content": "Explain error 0x0003"},
        {"role": "assistant", "content": "0x0003 - Fire aborted"},
        {"role": "user", "content": "Is it severe?"},
        {"role": "assistant", "content": "No severity listed."}
    ]

    has_followup, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What is the solution?", history
    )
    assert has_followup is True
    assert code == "0x0003"
    assert exp == "What is the solution? 0x0003"


def test_case_a_greeting_neutral_turn(session_id):
    """Case A: Explain error 0x0003 -> How do I fix it? -> Hi -> What does it mean?"""
    query("Explain error 0x0003", session_id=session_id)
    query("How do I fix it?", session_id=session_id)
    query("Hi", session_id=session_id)

    store = get_session_store()
    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What does it mean?", history
    )
    assert has_fu is True
    assert code == "0x0003"
    assert neutral_det is True
    assert skipped_count == 1
    assert source == "fault_code_history"
    assert exp == "What does it mean? 0x0003"


def test_case_b_thanks_neutral_turn(session_id):
    """Case B: Explain error 0x0003 -> Thanks -> What is the remedy?"""
    query("Explain error 0x0003", session_id=session_id)
    query("Thanks", session_id=session_id)

    store = get_session_store()
    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What is the remedy?", history
    )
    assert has_fu is True
    assert code == "0x0003"
    assert neutral_det is True
    assert skipped_count == 1
    assert source == "fault_code_history"
    assert exp == "What is the remedy? 0x0003"


def test_case_c_greeting_followed_by_topic_shift(session_id):
    """
    Case C: Explain error 0x0003 -> Hello -> How do I start a bike? -> What is the first step?

    Validates that:
    - A neutral greeting (Hello) is skipped during context resolution.
    - A subsequent non-fault-code topic resets context to general_topic.
    - The follow-up query does NOT resolve to 0x0003.

    The bike turn is injected directly to eliminate retrieval/embedding/Ollama dependency.
    """
    query("Explain error 0x0003", session_id=session_id)
    query("Hello", session_id=session_id)
    # Inject the topic-shift turn directly — session-memory logic only cares
    # that the turn exists in history, not how it arrived.
    store = get_session_store()
    store.add_turn(
        session_id,
        "How do I start a bike?",
        "To start a bike, first ensure the side stand is up.",
    )

    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What is the first step?", history
    )
    assert code is None
    assert source == "general_topic"
    assert exp == "What is the first step?"


def test_case_d_multiple_consecutive_neutral_turns(session_id):
    """Case D: Explain error 0x0003 -> Okay -> Okay -> Okay -> What does it mean?"""
    query("Explain error 0x0003", session_id=session_id)
    query("Okay", session_id=session_id)
    query("Okay", session_id=session_id)
    query("Okay", session_id=session_id)

    store = get_session_store()
    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What does it mean?", history
    )
    assert has_fu is True
    assert code == "0x0003"
    assert neutral_det is True
    assert skipped_count == 3
    assert source == "fault_code_history"


def test_case_e_mixed_neutral_turns(session_id):
    """Case E: Explain error 0x0003 -> Hi -> Hello -> Thanks -> What does it mean?"""
    query("Explain error 0x0003", session_id=session_id)
    query("Hi", session_id=session_id)
    query("Hello", session_id=session_id)
    query("Thanks", session_id=session_id)

    store = get_session_store()
    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What does it mean?", history
    )
    assert has_fu is True
    assert code == "0x0003"
    assert neutral_det is True
    assert skipped_count == 3
    assert source == "fault_code_history"


def test_case_f_neutral_turns_interspersed_with_topic_shift(session_id):
    """
    Case F: Explain error 0x0003 -> Hi -> How do I start a bike? -> Thanks -> What is the first step?

    Validates that:
    - Neutral turns (Hi, Thanks) are skipped during context resolution.
    - A non-fault-code topic between neutral turns resets context to general_topic.
    - The follow-up query does NOT resolve to 0x0003.

    The bike turn is injected directly to eliminate retrieval/embedding/Ollama dependency.
    """
    query("Explain error 0x0003", session_id=session_id)
    query("Hi", session_id=session_id)
    # Inject the topic-shift turn directly — session-memory logic only cares
    # that the turn exists in history, not how it arrived.
    store = get_session_store()
    store.add_turn(
        session_id,
        "How do I start a bike?",
        "To start a bike, first ensure the side stand is up.",
    )
    query("Thanks", session_id=session_id)

    history = store.get_history(session_id)
    has_fu, code, exp, neutral_det, skipped_count, source = _resolve_history_context(
        "What is the first step?", history
    )
    assert code is None
    assert source == "general_topic"
    assert exp == "What is the first step?"


def test_case_g_false_positive_protection_compound_question():
    """Case G: 'Hello, what does error 0x0003 mean?' should NOT be treated as a neutral turn."""
    from app.rag.pipeline import _is_neutral_conversation_turn
    assert _is_neutral_conversation_turn("Hello, what does error 0x0003 mean?") is False


def test_case_h_false_positive_protection_compound_instruction():
    """Case H: 'Thanks, now explain the remedy.' should NOT be treated as a neutral turn."""
    from app.rag.pipeline import _is_neutral_conversation_turn
    assert _is_neutral_conversation_turn("Thanks, now explain the remedy.") is False


def test_get_conversational_response_unit():
    """Test _get_conversational_response helper for all category mappings and false positive protection."""
    from app.rag.pipeline import _get_conversational_response

    # Greetings
    assert _get_conversational_response("hi") == "Hello! How can I help you today?"
    assert _get_conversational_response("Hello") == "Hello! How can I help you today?"
    assert _get_conversational_response("HEY") == "Hello! How can I help you today?"

    # Morning / Evening
    assert _get_conversational_response("good morning") == "Good morning! How can I assist you today?"
    assert _get_conversational_response("Good Evening") == "Good evening! How can I assist you today?"

    # Thanks
    assert _get_conversational_response("thanks") == "You're welcome."
    assert _get_conversational_response("Thank You") == "You're welcome."

    # Acknowledgements / Confirmations
    assert _get_conversational_response("ok") == "Got it."
    assert _get_conversational_response("okay") == "Got it."
    assert _get_conversational_response("Alright") == "Got it."
    assert _get_conversational_response("cool") == "Got it."
    assert _get_conversational_response("got it") == "Got it."
    assert _get_conversational_response("understood") == "Got it."
    assert _get_conversational_response("makes sense") == "Got it."

    # Farewell
    assert _get_conversational_response("bye") == "Goodbye! Have a great day."
    assert _get_conversational_response("Goodbye") == "Goodbye! Have a great day."

    # False positive protection — should return None
    assert _get_conversational_response("Hello?") is None
    assert _get_conversational_response("Hi, how do I fix 0x0003?") is None
    assert _get_conversational_response("Thanks, now explain the remedy.") is None
    assert _get_conversational_response("Explain error 0x0003") is None


def test_conversational_shortcut_e2e_sequence_a(session_id):
    """Verify Sequence A: Explain error 0x0003 -> Hi (shortcut) -> What does it mean? (resolves 0x0003)"""
    res1 = query("Explain error 0x0003", session_id=session_id)
    assert res1.found is True
    assert "0x0003" in res1.answer

    # Hi should bypass RAG/LLM and return conversational response immediately
    res2 = query("Hi", session_id=session_id)
    assert res2.answer == "Hello! How can I help you today?"
    assert res2.retrieved_chunks == []
    assert res2.top_score == 1.0

    # Follow-up should still resolve 0x0003
    res3 = query("What does it mean?", session_id=session_id)
    assert res3.found is True
    assert "0x0003" in res3.answer


def test_conversational_shortcut_e2e_sequence_b(session_id):
    """Verify Sequence B: Explain error 0x0003 -> Thanks (shortcut) -> What is the remedy? (resolves 0x0003)"""
    query("Explain error 0x0003", session_id=session_id)

    res2 = query("Thanks", session_id=session_id)
    assert res2.answer == "You're welcome."
    assert res2.retrieved_chunks == []
    assert res2.top_score == 1.0

    res3 = query("What is the remedy?", session_id=session_id)
    assert res3.found is True
    assert "The available documentation does not provide corrective steps for error 0x0003." in res3.answer
