import threading
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

import faiss  # type: ignore
import numpy as np

from app.rag.retriever import Retriever
from app.ingestion.ingest import run_ingestion
from app import settings


def test_mismatched_index_chunks_raises_value_error(tmp_path):
    """(b) Simulate a corrupted pair, confirm it raises rather than serving wrong data."""
    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.jsonl"

    # Create an index with 3 vectors
    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(3, dim).astype("float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    # Create chunks list with only 2 chunks (mismatched!) (JSONL format, 1 chunk per line)
    chunks_content = '{"chunk_id": "1"}\n{"chunk_id": "2"}\n'
    chunks_path.write_text(chunks_content, encoding="utf-8")

    # Trying to initialize should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        Retriever(index_path=index_path, chunks_path=chunks_path)
    assert "Fidelity violation" in str(excinfo.value)


def test_concurrent_reload_and_retrieve(tmp_path):
    """(a) Concurrent reload() + retrieve() must not raise exceptions."""
    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.jsonl"

    dim = 384
    index = faiss.IndexFlatIP(dim)
    vectors = np.random.rand(2, dim).astype("float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    # Correct number of chunks (2 chunks, JSONL format)
    chunks_content = '{"chunk_id": "1", "error_code": "0x0001"}\n{"chunk_id": "2", "error_code": "0x0002"}\n'
    chunks_path.write_text(chunks_content, encoding="utf-8")

    retriever = Retriever(index_path=index_path, chunks_path=chunks_path)

    # We need to temporarily patch standard settings paths so reload() loads from our tmp_path
    with patch("app.rag.retriever.FAISS_INDEX_PATH", index_path), \
            patch("app.rag.retriever.CHUNKS_STORE_PATH", chunks_path):

        errors = []

        def do_retrievals():
            for _ in range(50):
                try:
                    # Exact lookup or semantic search
                    retriever.retrieve("0x0001")
                except Exception as e:
                    errors.append(e)

        def do_reloads():
            for _ in range(10):
                try:
                    retriever.reload()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=do_retrievals),
            threading.Thread(target=do_retrievals),
            threading.Thread(target=do_reloads),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered concurrent errors: {errors}"


def test_atomic_swap_rollback_on_failure(tmp_path, monkeypatch):
    """(c) Simulate the second move failing, confirm the original pair is left intact."""
    # Setup standard files paths to point to tmp_path
    original_chunks_path = tmp_path / "chunks.jsonl"
    original_index_path = tmp_path / "faiss.index"

    # Write initial/original contents
    original_chunks_path.write_text("original chunks", encoding="utf-8")
    original_index_path.write_text("original index", encoding="utf-8")

    # Patch paths in ingest settings
    monkeypatch.setattr("app.ingestion.ingest.CHUNKS_STORE_PATH", original_chunks_path)
    monkeypatch.setattr("app.ingestion.ingest.FAISS_INDEX_PATH", original_index_path)

    # Let's mock _embed_chunks, parse_pdf, validate_chunks, _build_faiss_index, and save_chunks
    # so we don't do real ingestion but trigger the swap.
    monkeypatch.setattr("app.ingestion.ingest.parse_pdf", lambda path: {"rows": [], "source_hash": "123"})
    monkeypatch.setattr("app.ingestion.ingest.build_chunks", lambda res, name: [{"chunk_id": "1"}])
    monkeypatch.setattr("app.ingestion.ingest.validate_chunks", lambda chunks, res: [])
    monkeypatch.setattr("app.ingestion.ingest._embed_chunks", lambda chunks: np.zeros((1, 384), dtype=np.float32))
    monkeypatch.setattr("app.ingestion.ingest._build_faiss_index", lambda vecs: faiss.IndexFlatIP(384))
    monkeypatch.setattr("app.ingestion.ingest.save_chunks", lambda chunks, path: path.write_text("new chunks"))
    monkeypatch.setattr("app.ingestion.ingest._save_faiss_index", lambda index, path: path.write_text("new index"))

    # Now, let's mock shutil.move to fail specifically on moving the index file (the second move)
    original_move = shutil.move

    def mock_move(src, dst):
        if "faiss.index" in str(dst) and "faiss.tmp" in str(src):
            raise IOError("Simulated disk error during index move!")
        return original_move(src, dst)

    monkeypatch.setattr(shutil, "move", mock_move)

    # Run ingestion
    success = run_ingestion(Path("dummy.pdf"))

    # Assert that ingestion failed
    assert not success

    # Assert that original files are intact
    assert original_chunks_path.read_text(encoding="utf-8") == "original chunks"
    assert original_index_path.read_text(encoding="utf-8") == "original index"

    # Assert backup files are cleaned up or not present
    bak_chunks = original_chunks_path.with_suffix(".bak")
    bak_index = original_index_path.with_suffix(".bak")
    assert not bak_chunks.exists()
    assert not bak_index.exists()
