"""Shared configuration loader — reads app/config.yaml once at import."""
import yaml
from pathlib import Path

# Resolve config relative to this file's location (app/)
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Single global config dict — imported by all modules
CFG: dict = _load()

# ── Convenience accessors ──────────────────────────────────────────────────
MODEL_TAG: str = CFG["model"]["ollama_tag"]
OLLAMA_URL: str = CFG["model"]["ollama_url"]
TEMPERATURE: float = CFG["model"]["temperature"]
MAX_TOKENS: int = CFG["model"]["max_tokens"]
TIMEOUT: int = CFG["model"]["timeout_seconds"]

EMBED_MODEL: str = CFG["embedding"]["model_name"]
EMBED_DEVICE: str = CFG["embedding"]["device"]
QUERY_PREFIX: str = CFG["embedding"]["query_prefix"]

TOP_K: int = CFG["retrieval"]["top_k"]
CONFIDENCE_THRESHOLD: float = CFG["retrieval"]["confidence_threshold"]
RETURN_N: int = CFG["retrieval"]["return_n"]
ERROR_CODE_PATTERN: str = CFG["retrieval"]["error_code_pattern"]

# Paths are relative to the project root (one level up from app/)
_ROOT = Path(__file__).parent.parent
FAISS_INDEX_PATH: Path = _ROOT / CFG["paths"]["faiss_index"]
CHUNKS_STORE_PATH: Path = _ROOT / CFG["paths"]["chunks_store"]
SESSION_DB_PATH: Path = _ROOT / CFG["paths"]["session_db"]
LOG_FILE_PATH: Path = _ROOT / CFG["paths"]["log_file"]
SOURCE_PDF_PATH: Path = _ROOT / CFG["paths"]["source_pdf"]

SERVER_HOST: str = CFG["server"]["host"]
SERVER_PORT: int = CFG["server"]["port"]

LOG_LEVEL: str = CFG["logging"]["level"]
LOG_MAX_BYTES: int = CFG["logging"]["max_bytes"]
LOG_BACKUP_COUNT: int = CFG["logging"]["backup_count"]

MAX_HISTORY_TURNS: int = CFG.get("session", {}).get("max_history_turns", 4)
MAX_STORED_MESSAGES: int = CFG.get("session", {}).get("max_stored_messages", 100)
MAX_MESSAGE_CHARS: int = CFG.get("session", {}).get("max_message_chars", 300)
