"""CCMC Flare Scoreboard ingest package: discover, download, parse, save."""

from flare_scoreboard.config import load_config
from flare_scoreboard.crawl import discover_models
from flare_scoreboard.http_client import normalize_dir
from flare_scoreboard.pipeline import process_model, process_one

__all__ = [
    "load_config",
    "normalize_dir",
    "discover_models",
    "process_model",
    "process_one",
]
