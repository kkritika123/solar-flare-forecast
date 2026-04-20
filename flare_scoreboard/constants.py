"""Network defaults and the CSV column order used in every output file."""
USER_AGENT = "Mozilla/5.0 (flare-scoreboard-pipeline/2.4)"
TIMEOUT = 60

FIELDNAMES = [
    "model_name",
    "method",
    "forecast_type",
    "issue_time_utc",
    "runtime_utc",
    "window_begin_utc",
    "window_end_utc",
    "flare_threshold",
    "probability",
    "region_id",
    "source_url",
    "source_file",
    "year",
]
