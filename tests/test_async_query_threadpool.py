import asyncio
import time
from unittest.mock import patch
import pytest
import httpx

from app.main import app
from app.rag.pipeline import QueryResponse as PipelineResponse
from app.settings import SERVER_PORT


@pytest.mark.anyio
async def test_query_threadpool_does_not_block_health():
    """
    (C-1) POST /query is plain def, so FastAPI runs it in a threadpool.
    This ensures that GET /health still responds promptly while a query is in flight.
    """
    same_origin = f"http://127.0.0.1:{SERVER_PORT}"

    # Mock a slow pipeline call (sleeping for 1.0 second)
    def slow_query(*args, **kwargs):
        time.sleep(1.0)
        return PipelineResponse(
            answer="Delayed answer",
            citations=[],
            retrieved_chunks=[],
            top_score=1.0,
            latency_ms=1000,
            found=True,
        )

    # Mock health's httpx client get so it doesn't block on real network calls to non-running Ollama
    async def mock_httpx_get(*args, **kwargs):
        class MockResponse:
            status_code = 200

            def json(self):
                return {"models": [{"name": "qwen2.5:3b"}]}
        return MockResponse()

    with patch("app.main.rag_query", side_effect=slow_query), \
            patch("httpx.AsyncClient.get", side_effect=mock_httpx_get):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

            # Start the slow /query POST request
            async def trigger_query():
                t0 = time.perf_counter()
                response = await client.post(
                    "/query",
                    json={"question": "What does 0x0003 mean?"},
                    headers={"Origin": same_origin},
                )
                elapsed = time.perf_counter() - t0
                return response, elapsed

            # Start the query task
            query_task = asyncio.create_task(trigger_query())

            # Sleep briefly to ensure the query task actually enters the handler/thread
            await asyncio.sleep(0.2)

            # Call /health, which should respond immediately (way before 1.0 second)
            t_health_start = time.perf_counter()
            health_response = await client.get("/health")
            health_elapsed = time.perf_counter() - t_health_start

            # Wait for the query task to complete
            query_response, query_elapsed = await query_task

            # Assert health responded promptly (e.g. in less than 200ms)
            assert health_response.status_code in (200, 503)
            assert health_elapsed < 0.2, f"Health request was blocked! Took {health_elapsed:.3f}s"

            # Assert query completed successfully and took around 1 second
            assert query_response.status_code == 200
            assert query_elapsed >= 1.0
