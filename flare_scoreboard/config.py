"""Load and normalize config.json for the pipeline."""
import json
from typing import Any, Dict


def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["years_set"] = {int(y) for y in cfg["years"]}
    cfg["parse_exts_set"] = {
        str(e).lower() for e in cfg.get("parse_exts", ["xml", "txt", "json"])
    }
    cfg["workers"] = int(cfg.get("workers", 6))

    val = cfg.get("download_all_files", False)
    if isinstance(val, str):
        cfg["download_all_files"] = val.strip().lower() == "true"
    else:
        cfg["download_all_files"] = bool(val)

    cfg["out_dir"] = cfg.get("out_dir", "output")
    cfg["raw_dir"] = cfg.get("raw_dir", "data_raw")

    # Optional: only process folders whose .txt use the ASSA/AMOS tabular layout
    # (#Full Disk Forecast / #Region Forecast). If non-empty, overrides "models".
    assa = cfg.get("assa_format_models")
    m = cfg.get("models")

    if isinstance(assa, list) and len(assa) > 0:
        cfg["models_filter"] = {str(x).strip() for x in assa if str(x).strip()}
        cfg["models_filter_label"] = "assa_format_models"
    elif m is None:
        cfg["models_filter"] = None
        cfg["models_filter_label"] = None
    else:
        cfg["models_filter"] = {str(x).strip() for x in m if str(x).strip()}
        cfg["models_filter_label"] = "models"

    # Evaluation (model.py): hours padded on each side of [window_begin, window_end].
    cfg["event_window_match_tolerance_hours"] = float(
        cfg.get("event_window_match_tolerance_hours", 2.0)
    )

    # When both window times are missing (common for plain .txt scoreboard files), set
    # window_begin = issue_time_utc and window_end = issue + N hours (e.g. 24).
    fw = cfg.get("forecast_window_fill_hours")
    if fw is None or fw == "":
        cfg["forecast_window_fill_hours"] = None
    else:
        cfg["forecast_window_fill_hours"] = float(fw)

    return cfg
