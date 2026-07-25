import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.session import SessionStore



class TestSessionStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_session.db"
        self.store = SessionStore(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_init_and_empty_history(self):
        history = self.store.get_history("non_existent_session")
        self.assertEqual(history, [])

    def test_add_and_retrieve_history(self):
        session_id = "test_sess_1"
        self.store.add_turn(
            session_id,
            user_question="Explain error 0x0003",
            assistant_answer="0x0003 is High Voltage Output Failure.",
        )

        history = self.store.get_history(session_id, max_turns=4)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Explain error 0x0003")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "0x0003 is High Voltage Output Failure.")

    def test_history_truncation_to_max_turns(self):
        session_id = "test_sess_turns"
        for i in range(10):
            self.store.add_turn(session_id, f"Question {i}", f"Answer {i}")

        # Fetch max 4 turns = 8 messages
        history = self.store.get_history(session_id, max_turns=4)
        self.assertEqual(len(history), 8)
        # Check chronological order (latest 4 turns: index 6..9)
        self.assertEqual(history[0]["content"], "Question 6")
        self.assertEqual(history[1]["content"], "Answer 6")
        self.assertEqual(history[-2]["content"], "Question 9")
        self.assertEqual(history[-1]["content"], "Answer 9")

    def test_auto_pruning_stored_messages(self):
        session_id = "test_sess_prune"
        # Insert 12 messages with max_stored = 6
        for i in range(6):
            self.store.add_turn(session_id, f"Q{i}", f"A{i}", max_stored=6)

        # Database should only contain the latest 6 messages
        conn = self.store._get_connection()
        try:
            cursor = conn.execute(
                "SELECT count(*) FROM messages WHERE session_id = ?", (session_id,)
            )
            count = cursor.fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(count, 6)


        # Verify content of remaining 6 messages (Q3, A3, Q4, A4, Q5, A5)
        history = self.store.get_history(session_id, max_turns=10)
        self.assertEqual(len(history), 6)
        self.assertEqual(history[0]["content"], "Q3")
        self.assertEqual(history[-1]["content"], "A5")

    def test_clear_session(self):
        session_id = "test_sess_clear"
        self.store.add_turn(session_id, "Hello", "Hi")
        self.assertEqual(len(self.store.get_history(session_id)), 2)

        self.store.clear_session(session_id)
        self.assertEqual(len(self.store.get_history(session_id)), 0)


if __name__ == "__main__":
    unittest.main()
