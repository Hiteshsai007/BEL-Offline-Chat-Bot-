"""Shared logger — import this in every module instead of calling logging.basicConfig."""
import logging
import sys
from logging.handlers import RotatingFileHandler

from app.settings import LOG_FILE_PATH, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with both file and stderr handlers."""
    parent = logging.getLogger("bel")
    if not parent.handlers:
        parent.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler (rotating, local-only per Section 12)
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        parent.addHandler(fh)

        # Console handler (stderr so it doesn't pollute stdout)
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
        parent.addHandler(ch)

    if not name.startswith("bel"):
        full_name = f"bel.{name}"
    else:
        full_name = name

    return logging.getLogger(full_name)
