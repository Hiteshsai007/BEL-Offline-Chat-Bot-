"""Shared configuration loader — reads app/config.yaml once at import."""
from pathlib import Path

import yaml

# Resolve config relative to this file's location (app/)
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _get(cfg: dict, *path: str):
    """
    Traverse nested dicts; raise a descriptive error on missing keys.

    A bare KeyError at import time is impossible to diagnose because the
    traceback does not name the config file.  This helper produces a message
    like:

        Missing required config key: model -> ollama_tag (check app/config.yaml)
    """
    for key in path:
        if not isinstance(cfg, dict):
            raise TypeError(
                f"Config path {' -> '.join(path)!r}: expected dict at "
                f"{key!r}, got {type(cfg).__name__}"
            )
        if key not in cfg:
            raise KeyError(
                f"Missing required config key: {' -> '.join(path)!r} "
                f"(check {_CONFIG_PATH})"
            )
        cfg = cfg[key]
    return cfg


def _load() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Config file not found: {_CONFIG_PATH}. "
            "Copy app/config.yaml.example or restore the file."
        ) from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Config file {_CONFIG_PATH} is not valid YAML: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise TypeError(
            f"Config file {_CONFIG_PATH} must contain a top-level mapping, "
            f"not {type(raw).__name__}"
        )
    return raw


# Single global config dict — imported by all modules
CFG: dict = _load()

# ── Convenience accessors ──────────────────────────────────────────────────
MODEL_TAG: str = _get(CFG, "model", "ollama_tag")
OLLAMA_URL: str = _get(CFG, "model", "ollama_url")
TEMPERATURE: float = _get(CFG, "model", "temperature")
MAX_TOKENS: int = _get(CFG, "model", "max_tokens")
TIMEOUT: int = _get(CFG, "model", "timeout_seconds")
NUM_CTX: int = CFG.get("model", {}).get("num_ctx", 2048)


EMBED_MODEL: str = _get(CFG, "embedding", "model_name")
EMBED_DEVICE: str = _get(CFG, "embedding", "device")
QUERY_PREFIX: str = _get(CFG, "embedding", "query_prefix")

TOP_K: int = _get(CFG, "retrieval", "top_k")
CONFIDENCE_THRESHOLD: float = _get(CFG, "retrieval", "confidence_threshold")
RETURN_N: int = _get(CFG, "retrieval", "return_n")
ERROR_CODE_PATTERN: str = _get(CFG, "retrieval", "error_code_pattern")
DIRECT_ANSWER_THRESHOLD: float = _get(CFG, "retrieval", "direct_answer_threshold")
BM25_ENABLED: bool = CFG.get("retrieval", {}).get("bm25_enabled", True)
RRF_K: int = CFG.get("retrieval", {}).get("rrf_k", 60)
RETRIEVAL_CANDIDATES: int = CFG.get("retrieval", {}).get("retrieval_candidates", 20)
RERANKER_ENABLED: bool = CFG.get("retrieval", {}).get("reranker_enabled", True)
RERANKER_MODEL: str = CFG.get("retrieval", {}).get("reranker_model", "BAAI/bge-reranker-base")


# Paths are relative to the project root (one level up from app/)
_ROOT = Path(__file__).parent.parent
FAISS_INDEX_PATH: Path = _ROOT / _get(CFG, "paths", "faiss_index")
CHUNKS_STORE_PATH: Path = _ROOT / _get(CFG, "paths", "chunks_store")
SESSION_DB_PATH: Path = _ROOT / _get(CFG, "paths", "session_db")
LOG_FILE_PATH: Path = _ROOT / _get(CFG, "paths", "log_file")
SOURCE_PDF_PATH: Path = _ROOT / _get(CFG, "paths", "source_pdf")

SERVER_HOST: str = _get(CFG, "server", "host")
SERVER_PORT: int = _get(CFG, "server", "port")

LOG_LEVEL: str = _get(CFG, "logging", "level")
LOG_MAX_BYTES: int = _get(CFG, "logging", "max_bytes")
LOG_BACKUP_COUNT: int = _get(CFG, "logging", "backup_count")

MAX_HISTORY_TURNS: int = CFG.get("session", {}).get("max_history_turns", 4)
MAX_STORED_MESSAGES: int = CFG.get("session", {}).get("max_stored_messages", 100)
MAX_MESSAGE_CHARS: int = CFG.get("session", {}).get("max_message_chars", 300)
