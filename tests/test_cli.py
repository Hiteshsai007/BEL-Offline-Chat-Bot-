"""
Tests for app/cli.py — actual query flow, not just import.

Mocks the RAG backend so the CLI can be exercised without FAISS or Ollama.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.cli import main


def test_cli_query_flow() -> None:
    """CLI must run a query, print the answer, and return cleanly."""
    with patch("app.cli.get_retriever"):
        with patch("app.cli.get_embedder"):
            with patch("app.cli.query") as mock_query:
                mock_result = MagicMock()
                mock_result.error = None
                mock_result.answer = "Fire aborted."
                mock_result.retrieved_chunks = [
                    {
                        "error_code": "0x0003",
                        "error_description": "Fire aborted",
                        "score": 1.0,
                    },
                ]
                mock_result.guardrail_triggered = False
                mock_result.latency_ms = 500
                mock_query.return_value = mock_result

                with patch(
                    "app.cli.Prompt.ask",
                    side_effect=["What is 0x0003?", "exit"],
                ):
                    with patch("app.cli.console.print"):
                        # Should return cleanly (no exception)
                        main()

                mock_query.assert_called_once_with("What is 0x0003?")


def test_cli_startup_error() -> None:
    """Startup failure (e.g. missing index) must exit with code 1."""
    with patch(
        "app.cli.get_retriever",
        side_effect=Exception("Index not found"),
    ):
        with patch("app.cli.console.print"):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 1
