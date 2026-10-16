
"""Persistent per-file checkpoints for forecast parsing."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
CHECKPOINT_VERSION = 1

def checkpoint_path(
    url: str,
    cache_dir: str,
    model_name: str = "",
) -> Path:
    """Generate a model-specific checkpoint filename."""
    key = f"{model_name}:{url}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def load_checkpoint(
    url: str,
    cache_dir: str,
    source_path: str,
    model_name: str = "",
) -> Optional[List[Dict]]:
    """Load cached rows if the source file has not changed."""
    path = checkpoint_path(url, cache_dir, model_name)

    if not path.is_file() or not os.path.isfile(source_path):
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("checkpoint_version") != CHECKPOINT_VERSION:
            return None

        stat = os.stat(source_path)

        if data.get("source_size") != stat.st_size:
            return None

        if data.get("source_mtime_ns") != stat.st_mtime_ns:
            return None

        rows = data.get("rows")

        if not isinstance(rows, list):
            return None

        return rows

    except (OSError, ValueError, TypeError):
        return None


def save_checkpoint(
    url: str,
    cache_dir: str,
    source_path: str,
    rows: List[Dict],
    model_name: str = "",
) -> None:
    """Atomically save parsed rows and source metadata."""
    path = checkpoint_path(url, cache_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    stat = os.stat(source_path)

    data = {
    "checkpoint_version": CHECKPOINT_VERSION,
    "source_size": stat.st_size,
    "source_mtime_ns": stat.st_mtime_ns,
    "rows": rows,
    }

    temporary = path.with_suffix(".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file)

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
