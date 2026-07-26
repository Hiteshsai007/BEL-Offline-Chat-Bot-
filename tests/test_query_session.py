"""
Integration & unit tests for FastAPI /query endpoint with Session Conversation History.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


class TestQuerySessionEndpoint(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_session.db"
        self.patcher = patch("app.session.SESSION_DB_PATH", self.db_path)
        self.patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    @patch("app.rag.pipeline.get_retriever")
    def test_query_without_session_id_backward_compatibility(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_chunk = {
            "error_code": "0x0003",
            "error_description": "High Voltage Output Failure",
            "error_remarks": "Check fuse",
            "document_name": "IRL Fault Codes",
        }
        mock_retriever.retrieve.return_value = [
            MagicMock(chunk=mock_chunk, score=1.0)
        ]
        mock_get_retriever.return_value = mock_retriever

        # Send request without session_id
        resp = self.client.post("/query", json={"question": "What does 0x0003 mean?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["found"])
        self.assertIn("0x0003", data["answer"])
        self.assertIsNotNone(data.get("session_id"))
        self.assertTrue(len(data["session_id"]) > 0)

    @patch("app.rag.pipeline.get_retriever")
    def test_query_multiturn_flow_and_session_recording(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_chunk = {
            "error_code": "0x0003",
            "error_description": "High Voltage Output Failure",
            "error_remarks": "Check fuse",
            "document_name": "IRL Fault Codes",
        }
        mock_retriever.retrieve.return_value = [
            MagicMock(chunk=mock_chunk, score=1.0)
        ]
        mock_get_retriever.return_value = mock_retriever

        session_id = "test_multiturn_123"

        # Turn 1
        resp1 = self.client.post(
            "/query",
            json={"question": "Explain error 0x0003", "session_id": session_id},
        )
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp1.json()["session_id"], session_id)

        # Turn 2 (contextual follow-up)
        resp2 = self.client.post(
            "/query",
            json={"question": "How to fix it?", "session_id": session_id},
        )
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json()["session_id"], session_id)

        # Verify retriever was called with context-expanded query for Turn 2
        calls = mock_retriever.retrieve.call_args_list
        self.assertTrue(len(calls) >= 2)
        second_call_arg = calls[1][0][0]
        # Should contain "0x0003" expanded from history
        self.assertIn("0x0003", second_call_arg)

    def test_clear_session_endpoint(self):
        session_id = "test_clear_endpoint_session"

        # Clear session
        resp = self.client.post("/session/clear", json={"session_id": session_id})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    @patch("app.rag.pipeline.get_retriever")
    def test_fresh_question_not_expanded_with_history_code(self, mock_get_retriever):
        """
        A fresh question without pronouns (e.g. 'How do I check tire pressure')
        must NOT be expanded with a fault code from previous conversation history.
        Only queries with explicit anaphoric references (pronouns like 'it', 'this',
        'the error') should inherit codes from history.
        """
        mock_retriever = MagicMock()
        mock_chunk = {
            "error_code": None,
            "error_description": "Tire pressure inspection",
            "error_remarks": "Check when cold",
            "document_name": "Kawasaki Manual",
        }
        mock_retriever.retrieve.return_value = [
            MagicMock(chunk=mock_chunk, score=0.75)
        ]
        mock_get_retriever.return_value = mock_retriever

        session_id = "test_no_expansion_123"

        # Turn 1: establish a fault code in history
        resp1 = self.client.post(
            "/query",
            json={"question": "What does 0x0027 mean?", "session_id": session_id},
        )
        self.assertEqual(resp1.status_code, 200)

        # Turn 2: fresh question, no pronouns — should NOT get 0x0027 appended
        resp2 = self.client.post(
            "/query",
            json={"question": "How do I check tire pressure", "session_id": session_id},
        )
        self.assertEqual(resp2.status_code, 200)

        # Verify retriever was called with the ORIGINAL question, no code appended
        calls = mock_retriever.retrieve.call_args_list
        self.assertTrue(len(calls) >= 2)
        second_call_arg = calls[1][0][0]
        self.assertNotIn("0x0027", second_call_arg)
        self.assertEqual(second_call_arg, "How do I check tire pressure")


if __name__ == "__main__":
    unittest.main()
