"""
FastAPI application — serves the web UI and the /query endpoint.

Endpoints:
    GET  /          → index.html (single-page UI)
    POST /query     → RAG pipeline query
    GET  /health    → Ollama + index liveness check
    POST /reload    → hot-reload the FAISS index after re-ingestion
    POST /session/clear → clear session conversation history

All traffic is loopback-only (127.0.0.1 — PRD Section 12).
No CORS, no remote origins, no telemetry.

State-changing routes (POST /query, POST /reload) are additionally guarded by
a same-origin check — see app/security.py for the threat model (finding S-1).
"""
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.logger import get_logger
from app.rag.pipeline import query as rag_query
from app.security import verify_same_origin
from app.session import get_session_store
from app.settings import FAISS_INDEX_PATH, MODEL_TAG, OLLAMA_URL, SERVER_HOST, SERVER_PORT

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Module-level readiness flag set by lifespan (finding H-7)
_startup_ready = False


# ── Lifespan: warm up retriever & embedder on startup ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_ready
    _startup_ready = False
    log.info("BEL Offline AI Interface starting …")
    try:
        import asyncio

        from app.rag.retriever import get_retriever
        await asyncio.to_thread(get_retriever)  # load FAISS index + chunks in thread
        from app.rag.embedder import get_embedder
        await asyncio.to_thread(get_embedder)   # load BGE model on CPU in thread
        log.info("Retriever and embedder ready.")
        _startup_ready = True
    except FileNotFoundError as e:
        log.warning("Startup warning: %s", e)
        log.warning("Run ingestion first: python -m app.ingestion.ingest")
    except Exception as e:
        log.error("Startup error: %s", e)
    yield
    log.info("BEL Offline AI Interface shutting down.")


app = FastAPI(
    title="BEL Offline AI — Fault Code Lookup",
    version="1.0.0",
    docs_url=None,       # disable Swagger UI (not needed for this interface)
    redoc_url=None,
    lifespan=lifespan,
)

# Serve static files (CSS, JS) — all local, no CDN
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ───────────────────────────────────────────────
# Upper bound on an inbound question (finding S-3).
MAX_QUESTION_CHARS = 2000


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None


class ChunkInfo(BaseModel):
    error_code: str | None
    error_description: str | None
    error_remarks: str | None
    document_name: str | None
    page_number: int | None = None
    chunk_type: str | None = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunks: list[ChunkInfo]
    top_score: float
    latency_ms: int
    found: bool
    guardrail_triggered: bool
    error: str | None
    session_id: str | None = None


class ClearSessionRequest(BaseModel):
    session_id: str


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_path))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Serve the app favicon so browsers receive 200 instead of a noisy 404."""
    icon_path = STATIC_DIR / "favicon.ico"
    if not icon_path.exists():
        raise HTTPException(status_code=404, detail="favicon.ico not found")
    return FileResponse(str(icon_path), media_type="image/x-icon")


@app.get("/style.css", include_in_schema=False)
async def serve_css() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "style.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def serve_js() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "app.js"), media_type="application/javascript")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_same_origin)])
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Accept a plain-language question and return a grounded answer.

    The pipeline uses a hybrid lookup + LLM strategy: high-confidence
    retrieval results (exact code matches, strong semantic hits above
    DIRECT_ANSWER_THRESHOLD) are answered directly from source document
    metadata — deterministic, instant, zero-hallucination.  Lower-confidence
    queries are synthesised by the local LLM with citation guardrails.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    session_id = req.session_id.strip() if (req.session_id and req.session_id.strip()) else str(uuid.uuid4())
    log.info("POST /query | question=%r | session_id=%s", req.question[:80], session_id)
    result = rag_query(req.question, session_id=session_id)

    chunks_out = []
    for c in result.retrieved_chunks:
        chunks_out.append(ChunkInfo(
            error_code=c.get("error_code"),
            error_description=c.get("error_description"),
            error_remarks=c.get("error_remarks"),
            document_name=c.get("document_name"),
            page_number=c.get("page_number"),
            chunk_type=c.get("chunk_type"),
            score=c.get("score", 0.0),
        ))

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunks=chunks_out,
        top_score=round(result.top_score, 4),
        latency_ms=result.latency_ms,
        found=result.found,
        guardrail_triggered=result.guardrail_triggered,
        error=result.error,
        session_id=session_id,
    )


@app.post("/session/clear", dependencies=[Depends(verify_same_origin)])
async def clear_session_endpoint(req: ClearSessionRequest):
    if not req.session_id or not req.session_id.strip():
        raise HTTPException(status_code=422, detail="session_id must not be empty.")
    get_session_store().clear_session(req.session_id)
    return {"status": "ok", "message": f"Cleared history for session {req.session_id}"}


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness check — verifies Ollama is reachable and index exists."""
    from app.settings import get_active_model
    current_model = get_active_model()
    status = {
        "server": "ok",
        "index_exists": FAISS_INDEX_PATH.exists(),
        "ollama": "unknown",
        "model": current_model,
        "startup_ready": _startup_ready,
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                tags = [m["name"] for m in r.json().get("models", [])]
                status["ollama"] = "ok"
                status["model_pulled"] = any(current_model in t for t in tags)
                status["available_models"] = tags
            else:
                status["ollama"] = f"http_{r.status_code}"
    except Exception as e:
        log.error("Ollama health probe failed: %s", e)
        status["ollama"] = "unreachable"

    overall = (
        _startup_ready
        and status["index_exists"]
        and status["ollama"] == "ok"
        and status.get("model_pulled", False)
    )
    status["ready"] = overall
    code = 200 if overall else 503
    return JSONResponse(content=status, status_code=code)


@app.get("/models")
async def list_models() -> dict:
    """List available Ollama models."""
    from app.settings import OLLAMA_URL, get_active_model
    current = get_active_model()
    available = [current]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                tags = [m["name"] for m in r.json().get("models", [])]
                if tags:
                    available = tags
    except Exception as e:
        log.error("Failed to query Ollama tags: %s", e)
    return {"current": current, "available": available}


class SelectModelRequest(BaseModel):
    model: str


@app.post("/model/select", dependencies=[Depends(verify_same_origin)])
async def select_model(req: SelectModelRequest) -> dict:
    """Switch active Ollama model."""
    from app.settings import set_active_model
    if not req.model or not req.model.strip():
        raise HTTPException(status_code=422, detail="Model name must not be empty.")

    requested = req.model.strip()
    known = (await list_models())["available"]
    if requested not in known:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{requested}' is not installed in Ollama.",
        )

    new_model = set_active_model(requested)
    log.info("Active model switched to: %s", new_model)
    return {"status": "ok", "current_model": new_model}


@app.post("/reload", dependencies=[Depends(verify_same_origin)])
async def reload_index() -> dict:
    """Hot-reload the FAISS index after running the ingestion script."""
    try:
        from app.rag.retriever import get_retriever
        get_retriever().reload()
        return {"status": "ok", "message": "Index reloaded successfully."}
    except Exception as e:
        log.error("Reload failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Index reload failed. See the server log for details.",
        )


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level="info",
    )
