"""Configuration helpers for the flare scoreboard pipeline."""

import json
import os
from typing import Any, Dict, Optional


DEFAULT_CONFIG = {
    "base_url": (
        "https://iswa.ccmc.gsfc.nasa.gov/"
        "iswa_data_tree/model/solar/flare-scoreboard/"
    ),
    "years": [2020, 2021, 2022, 2023, 2024, 2025],
    "parse_exts": ["xml", "txt", "json"],
    "workers": 6,
    "raw_dir": "data_raw",
    "out_dir": "output",
    "download_all_files": False,
    "models": None,
    "assa_format_models": [],
    "event_window_match_tolerance_hours": 2,
    "forecast_window_fill_hours": 24,
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration.

    Uses built-in defaults so the installed package works without
    requiring a config.json file.

    If config.json exists in the current directory, its values override
    the defaults. A specific configuration file can also be supplied
    with the path argument.
    """

    cfg = DEFAULT_CONFIG.copy()

    config_path = path

    if config_path is None and os.path.isfile("config.json"):
        config_path = "config.json"

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)

        cfg.update(user_cfg)

    cfg["years_set"] = {int(y) for y in cfg["years"]}

    cfg["parse_exts_set"] = {
        str(e).lower()
        for e in cfg.get(
            "parse_exts",
            ["xml", "txt", "json"],
        )
    }

    cfg["workers"] = int(cfg.get("workers", 6))

    val = cfg.get("download_all_files", False)

    if isinstance(val, str):
        cfg["download_all_files"] = (
            val.strip().lower() == "true"
        )
    else:
        cfg["download_all_files"] = bool(val)

    cfg["out_dir"] = cfg.get("out_dir", "output")
    cfg["raw_dir"] = cfg.get("raw_dir", "data_raw")

    assa = cfg.get("assa_format_models")
    models = cfg.get("models")

    if isinstance(assa, list) and len(assa) > 0:
        cfg["models_filter"] = {
            str(x).strip()
            for x in assa
            if str(x).strip()
        }

        cfg["models_filter_label"] = "assa_format_models"

    elif models is None:
        cfg["models_filter"] = None
        cfg["models_filter_label"] = None

    else:
        cfg["models_filter"] = {
            str(x).strip()
            for x in models
            if str(x).strip()
        }

        cfg["models_filter_label"] = "models"

    cfg["event_window_match_tolerance_hours"] = float(
        cfg.get(
            "event_window_match_tolerance_hours",
            2.0,
        )
    )

    fw = cfg.get("forecast_window_fill_hours")

    if fw is None or fw == "":
        cfg["forecast_window_fill_hours"] = None
    else:
        cfg["forecast_window_fill_hours"] = float(fw)

    return cfg