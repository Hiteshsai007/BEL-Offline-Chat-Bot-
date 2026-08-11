"""Regression tests for indexes rebuilt by a separate ingestion process."""
import pickle
from unittest.mock import Mock

import app.rag.retriever as retriever_module
from app.rag.retriever import Retriever


class _FakeBM25:
    builds = 0

    def __init__(self, corpus):
        type(self).builds += 1
        self.corpus = corpus
        self.corpus_size = len(corpus)


def test_changed_index_files_trigger_automatic_reload(tmp_path):
    """The next query can notice an atomic replacement made by the CLI."""
    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.jsonl"
    general_index = tmp_path / "general" / "faiss.index"
    general_chunks = tmp_path / "general" / "chunks.jsonl"

    index_path.write_bytes(b"old-index")
    chunks_path.write_text("old chunks", encoding="utf-8")

    instance = Retriever.__new__(Retriever)
    instance._index_path = index_path
    instance._chunks_path = chunks_path
    instance._general_index_path = general_index
    instance._general_chunks_path = general_chunks
    instance._loaded_file_signature = instance._index_file_signature()
    instance.reload = Mock()

    # Simulate ingestion publishing a new general-document pair.
    general_index.parent.mkdir()
    general_index.write_bytes(b"new-index")
    general_chunks.write_text('{"chunk_text": "new document"}\n', encoding="utf-8")

    instance._reload_if_index_changed()

    instance.reload.assert_called_once_with()


def test_unchanged_index_files_do_not_reload(tmp_path):
    index_path = tmp_path / "faiss.index"
    chunks_path = tmp_path / "chunks.jsonl"
    index_path.write_bytes(b"index")
    chunks_path.write_text("chunks", encoding="utf-8")

    instance = Retriever.__new__(Retriever)
    instance._index_path = index_path
    instance._chunks_path = chunks_path
    instance._general_index_path = tmp_path / "general" / "faiss.index"
    instance._general_chunks_path = tmp_path / "general" / "chunks.jsonl"
    instance._loaded_file_signature = instance._index_file_signature()
    instance.reload = Mock()

    instance._reload_if_index_changed()

    instance.reload.assert_not_called()


def test_bm25_cache_is_bound_to_chunk_content(tmp_path, monkeypatch):
    """Equal chunk counts must not reuse keyword scores from an older doc."""
    _FakeBM25.builds = 0
    monkeypatch.setattr(retriever_module, "_bm25_available", True)
    monkeypatch.setattr(
        retriever_module, "BM25Okapi", _FakeBM25, raising=False
    )
    cache_path = tmp_path / "bm25.pkl"

    old_chunks = [{"chunk_id": "same", "chunk_text": "old manual content"}]
    new_chunks = [{"chunk_id": "same", "chunk_text": "replacement document data"}]

    old_bm25 = Retriever._build_bm25_index(old_chunks, cache_path)
    new_bm25 = Retriever._build_bm25_index(new_chunks, cache_path)

    assert _FakeBM25.builds == 2
    assert old_bm25.corpus != new_bm25.corpus
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    assert cached["version"] == 2
    assert cached["corpus_fingerprint"] == Retriever._chunks_fingerprint(new_chunks)
