#Simple public API for downloading CCMC Flare Scoreboard forecasts.

from flare_scoreboard.config import load_config
from flare_scoreboard.crawl import discover_models
from flare_scoreboard.http_client import normalize_dir
from flare_scoreboard.pipeline import process_model


def _model_name(model_url: str) -> str:
    """Return the model name from its directory URL."""
    return model_url.rstrip("/").split("/")[-1]


def download_forecasts(models=None, years=None):
    """
    Download and parse CCMC Flare Scoreboard forecasts.

    Parameters
    ----------
    models : list[str] or None
        Model names to download, for example ["NOAA_1", "SIDC_v2"].
        If None, models are taken from config.json.

    years : list[int] or None
        Years to download, for example [2024, 2025].
        If None, years are taken from config.json.
    """

    cfg = load_config()

    base_url = normalize_dir(cfg["base_url"])

    # Use years supplied by the user, otherwise use config.json.
    if years is None:
        years_set = cfg["years_set"]
    else:
        years_set = set(years)

    parse_exts = cfg["parse_exts_set"]
    raw_dir = cfg["raw_dir"]
    out_dir = cfg["out_dir"]
    workers = cfg["workers"]
    download_all = cfg["download_all_files"]

    print("Discovering model folders...")
    discovered_models = discover_models(base_url)

    # Use models supplied by the user.
    if models is not None:
        requested = set(models)

        model_urls = [
            url
            for url in discovered_models
            if _model_name(url) in requested
        ]

        discovered_names = {
            _model_name(url) for url in discovered_models
        }

        missing = requested - discovered_names

        if missing:
            print(
                "[WARN] Models not found:",
                ", ".join(sorted(missing))
            )

    else:
        # Fall back to config.json model selection.
        models_filter = cfg.get("models_filter")

        if models_filter is None:
            model_urls = discovered_models
        else:
            model_urls = [
                url
                for url in discovered_models
                if _model_name(url) in models_filter
            ]

    print(f"Processing {len(model_urls)} model(s).")

    for model_url in model_urls:
        process_model(
            model_url=model_url,
            years_set=years_set,
            parse_exts=parse_exts,
            raw_dir=raw_dir,
            out_dir=out_dir,
            workers=workers,
            download_all=download_all,
        )

    print("\nDone.")

    return None
def evaluate(models=None, years=None):
    """
    Evaluate downloaded forecasts against LMSAL observed flare events.

    If the required LMSAL event CSV does not exist, it is downloaded
    automatically for the requested evaluation years.

    Parameters
    ----------
    models : list[str] or None
        Models to evaluate, for example ["NOAA_1"].
        If None, all downloaded models are evaluated.

    years : list[int] or None
        Years to evaluate, for example [2024].
        If None, years are taken from the configuration.

    Returns
    -------
    pandas.DataFrame or None
        Combined evaluation results containing TSS and HSS scores.
    """
    import os
    import pandas as pd

    from flare_scoreboard.scoring import (
        FORECAST_OUTPUT_DIR,
        _reorder_score_columns,
        discover_models_with_forecasts,
        evaluate_one_model,
    )
    from flare_scoreboard.evaluation import load_lmsal_events
    from flare_scoreboard.lmsal import download_lmsal_events

    cfg = load_config()

    if years is None:
        eval_years = sorted(int(y) for y in cfg["years_set"])
    else:
        eval_years = sorted(int(y) for y in years)

    if not eval_years:
        raise ValueError("At least one evaluation year is required.")

    start_year = min(eval_years)
    end_year = max(eval_years)

    # Store the LMSAL data for the requested evaluation range.
    lmsal_csv = os.path.join(
        "data",
        f"lmsal_events_{start_year}_{end_year}.csv",
    )

    # Download LMSAL observations automatically when needed.
    if not os.path.isfile(lmsal_csv):
        print(
            "LMSAL event data not found. "
            "Downloading observations..."
        )

        download_lmsal_events(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            out=lmsal_csv,
        )

    print(f"Loading LMSAL events from: {lmsal_csv}")

    events_df = load_lmsal_events(lmsal_csv)

    print("LMSAL rows loaded:", len(events_df))

    tol_hours = cfg["event_window_match_tolerance_hours"]
    window_fill_hours = cfg.get(
        "forecast_window_fill_hours"
    )

    if models is None:
        model_names = discover_models_with_forecasts(
            FORECAST_OUTPUT_DIR
        )
    else:
        model_names = list(models)

    if not model_names:
        print("No models to evaluate.")
        return None

    os.makedirs("evaluation_results", exist_ok=True)

    all_results = []

    for model_name in model_names:

        result = evaluate_one_model(
            model_name,
            events_df,
            tol_hours=tol_hours,
            eval_years=eval_years,
            window_fill_hours=window_fill_hours,
        )

        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No evaluation results were produced.")
        return None

    combined = _reorder_score_columns(
        pd.concat(
            all_results,
            ignore_index=True,
        )
    )

    return combined