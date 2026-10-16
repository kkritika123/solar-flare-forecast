
from flare_scoreboard.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_save_and_load(tmp_path):
    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>test</forecast>")

    url = "https://example.com/forecast.xml"
    cache = str(tmp_path / "checkpoints")
    rows = [{"model": "NOAA", "probability": 0.5}]

    save_checkpoint(url, cache, str(source), rows)

    result = load_checkpoint(url, cache, str(source))

    assert result == rows


def test_checkpoint_invalidated_after_source_change(tmp_path):
    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>old</forecast>")

    url = "https://example.com/forecast.xml"
    cache = str(tmp_path / "checkpoints")

    save_checkpoint(url, cache, str(source), [{"value": 1}])

    source.write_text("<forecast>changed data</forecast>")

    result = load_checkpoint(url, cache, str(source))

    assert result is None


def test_missing_checkpoint_returns_none(tmp_path):
    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>test</forecast>")

    result = load_checkpoint(
        "https://example.com/forecast.xml",
        str(tmp_path / "checkpoints"),
        str(source),
    )

    assert result is None


import json
from unittest.mock import patch

from flare_scoreboard.checkpoint import checkpoint_path


def test_checkpoint_invalidated_by_version(tmp_path):
    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>test</forecast>")

    url = "https://example.com/forecast.xml"
    cache = str(tmp_path / "checkpoints")
    rows = [{"value": 1}]

    save_checkpoint(url, cache, str(source), rows)

    path = checkpoint_path(url, cache)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    data["checkpoint_version"] = -1

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    assert load_checkpoint(url, cache, str(source)) is None
