import logging
import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"


def _parse_log_level(level_name: str) -> int:
    name = (level_name or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def _ensure_dir(path: str) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logging(
    *,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configure standard Python logging for the backend.

    Defaults:
    - Logs to stdout with a consistent format.
    - Optional rotating file output when RADARS_LOG_DIR/LOG_DIR is provided.

    Env vars:
    - RADARS_LOG_LEVEL / LOG_LEVEL
    - RADARS_LOG_DIR / LOG_DIR
    - RADARS_LOG_MAX_BYTES (default: 20MB)
    - RADARS_LOG_BACKUP_COUNT (default: 5)
    """
    resolved_level = _parse_log_level(level or os.getenv("RADARS_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO")
    resolved_log_dir = log_dir or os.getenv("RADARS_LOG_DIR") or os.getenv("LOG_DIR")

    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(resolved_level)
        return

    root.handlers.clear()
    root.setLevel(resolved_level)

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if resolved_log_dir:
        from logging.handlers import RotatingFileHandler

        max_bytes = int(os.getenv("RADARS_LOG_MAX_BYTES") or str(20 * 1024 * 1024))
        backup_count = int(os.getenv("RADARS_LOG_BACKUP_COUNT") or "5")

        p = _ensure_dir(resolved_log_dir)
        file_handler = RotatingFileHandler(
            p / "radars-monitor.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

