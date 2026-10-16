
from flare_scoreboard.pipeline import filter_rows_by_date


def test_date_filter():
    rows = [
        {"issue_time_utc": "2024-01-14T12:00:00Z"},
        {"issue_time_utc": "2024-01-15T17:00:00Z"},
        {"issue_time_utc": "2024-02-09T17:00:00Z"},
        {"issue_time_utc": "2024-02-10T00:00:00Z"},
    ]

    result = filter_rows_by_date(
        rows,
        start="2024-01-15",
        end="2024-02-10",
    )

    assert len(result) == 2
    assert result[0]["issue_time_utc"] == "2024-01-15T17:00:00Z"
    assert result[1]["issue_time_utc"] == "2024-02-09T17:00:00Z"


def test_no_date_filter():
    rows = [
        {"issue_time_utc": "2023-01-01T00:00:00Z"},
        {"issue_time_utc": "2024-01-01T00:00:00Z"},
    ]

    assert filter_rows_by_date(rows) == rows


def test_invalid_date_range():
    import pytest

    with pytest.raises(ValueError):
        filter_rows_by_date(
            [],
            start="2024-02-10",
            end="2024-01-15",
        )
