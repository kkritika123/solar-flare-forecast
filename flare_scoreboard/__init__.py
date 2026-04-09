"""
NASA CCMC Flare Scoreboard ingest pipeline.

Layout (similar idea to a single well-sectioned app module):

- ``config`` — load ``config.json``
- ``http_client`` — session, directory listing, downloads
- ``crawl`` — discover models, find year folders, list files
- ``parsers`` — XML / TXT / JSON → row dicts
- ``csv_output`` — split rows into ``output/<MODEL>/<YEAR>_{full_disk,region}.csv``
- ``pipeline`` — ``process_model`` / ``process_one``
"""

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
