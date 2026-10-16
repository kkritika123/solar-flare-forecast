
from unittest.mock import patch

from flare_scoreboard.pipeline import process_one


def test_pipeline_reuses_checkpoint(tmp_path):
    url = "https://example.com/forecast.xml"
    model_name = "NOAA"
    raw_dir = str(tmp_path)

    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>test</forecast>")

    expected_rows = [
        {"model": "NOAA", "probability": 0.5}
    ]

    with (
        patch(
            "flare_scoreboard.pipeline.download_file",
            return_value=str(source),
        ),
        patch(
            "flare_scoreboard.pipeline.parse_iswa_xml",
            return_value=expected_rows,
        ) as parser,
    ):
        first_result = process_one(
            url, model_name, raw_dir
        )

        second_result = process_one(
            url, model_name, raw_dir
        )

    assert first_result == expected_rows
    assert second_result == expected_rows

    # Parsing should happen only once.
    assert parser.call_count == 1


def test_pipeline_reparses_changed_source(tmp_path):
    url = "https://example.com/forecast.xml"
    model_name = "NOAA"
    raw_dir = str(tmp_path)

    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>original</forecast>")

    with (
        patch(
            "flare_scoreboard.pipeline.download_file",
            return_value=str(source),
        ),
        patch(
            "flare_scoreboard.pipeline.parse_iswa_xml",
            return_value=[{"value": 1}],
        ) as parser,
    ):
        process_one(url, model_name, raw_dir)

        # Changing the source invalidates its checkpoint.
        source.write_text(
            "<forecast>updated with new data</forecast>"
        )

        process_one(url, model_name, raw_dir)

    assert parser.call_count == 2


def test_pipeline_does_not_checkpoint_empty_results(tmp_path):
    url = "https://example.com/forecast.xml"
    model_name = "NOAA"
    raw_dir = str(tmp_path)

    source = tmp_path / "forecast.xml"
    source.write_text("<forecast>empty</forecast>")

    with (
        patch(
            "flare_scoreboard.pipeline.download_file",
            return_value=str(source),
        ),
        patch(
            "flare_scoreboard.pipeline.parse_iswa_xml",
            return_value=[],
        ) as parser,
    ):
        process_one(url, model_name, raw_dir)
        process_one(url, model_name, raw_dir)

    # Empty results should not be checkpointed.
    assert parser.call_count == 2
