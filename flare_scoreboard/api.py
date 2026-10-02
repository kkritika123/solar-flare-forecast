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


def get_predictions(
    model_id,
    model_type,
    flare_type,
    start,
    end,
    source="hapi",
    archive_dir="output",
):
    """Retrieve forecasts through the HAPI API.

    start is inclusive; end is exclusive.
    flare_type is the exact HAPI field, such as M or MPlus.
    """
    from flare_scoreboard.hapi import download_hapi

    if source.lower() == "archive":
        return query_archive(
            model_id=model_id,
            model_type=model_type,
            flare_type=flare_type,
            start=start,
            end=end,
            archive_dir=archive_dir,
    )
    if source.lower() != "hapi":
        raise ValueError(
            "source must be 'hapi' or 'archive'."
        )

    df = download_hapi(
        model_id=model_id,
        model_type=model_type,
        flare_type=flare_type,
        start=start,
        end=end,
    )

    # Give the returned forecast fields consistent names.
    df = df.rename(
        columns={
            "issue_time": "issue_time_utc",
            "start_window": "window_begin_utc",
            "end_window": "window_end_utc",
            flare_type: "probability",
            "NOAARegionId": "noaa_region_id",
        }
    )

    df["model_id"] = model_id
    df["model_type"] = model_type
    df["flare_type"] = flare_type
    df["source"] = "hapi"

    return df

def query_archive(
    model_id,
    model_type,
    flare_type,
    start,
    end,
    archive_dir="output",
):
    """Select archive records by window start: start <= time < end."""
    from pathlib import Path

    import pandas as pd

    from flare_scoreboard.hapi import _utc_time

    types = {
        "full_disk": "full_disk",
        "active_region": "region",
    }

    if model_type not in types:
        raise ValueError("Use full_disk or active_region.")

    start_time = _utc_time(start)
    end_time = _utc_time(end)

    if start_time >= end_time:
        raise ValueError("start must be earlier than end.")

    folder = Path(archive_dir) / model_id
    files = sorted(folder.glob(f"*_{types[model_type]}.csv"))

    if not files:
        raise FileNotFoundError(
            f"No {model_type} archive CSVs found in {folder}"
        )

    df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True,
    )

    window_start = pd.to_datetime(
        df["window_begin_utc"],
        utc=True,
        format="mixed",
        errors="coerce",
)

    invalid_windows = window_start.isna()

    if invalid_windows.any():
        print(
            f"WARNING: {model_id}: "
            f"{invalid_windows.sum()} archive rows have missing or invalid "
            "window starts and cannot be assigned to a date range."
        )

    df["window_begin_utc"] = window_start

    df = df.loc[
        (df["model_name"] == model_id)
        & (df["forecast_type"] == types[model_type])
        & (df["flare_threshold"] == flare_type)
        & (df["window_begin_utc"] >= start_time)
        & (df["window_begin_utc"] < end_time)
    ].copy()

# Validate the remaining timestamps in the selected records.
    for column in ("issue_time_utc", "window_end_utc"):
        df[column] = pd.to_datetime(
            df[column],
            utc=True,
            format="mixed",
            errors="raise",
    )

    df["probability"] = pd.to_numeric(
        df["probability"], errors="coerce"
    )
    df = df.rename(columns={"region_id": "noaa_region_id"})
    df["model_id"] = model_id
    df["model_type"] = model_type
    df["flare_type"] = flare_type
    df["source"] = "archive"

    return df.reset_index(drop=True)

def compare_sources(
    model_id,
    model_type,
    flare_type,
    start,
    end,
    archive_dir="output",
    tolerance=1e-6,
):
    """Compare matching HAPI and processed archive forecasts."""
    import math
    import numpy as np
    import pandas as pd

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")

    query = dict(
        model_id=model_id,
        model_type=model_type,
        flare_type=flare_type,
        start=start,
        end=end,
    )

    hapi = get_predictions(**query, source="hapi")
    archive = get_predictions(
        **query,
        source="archive",
        archive_dir=archive_dir,
    )

    keys = [
        "window_begin_utc",
        "window_end_utc",
        "issue_time_utc",
    ]

    if model_type == "active_region":
        keys.append("noaa_region_id")

    def prepare(df, source):
        frame = df[keys + ["probability"]].copy()

        for column in keys:
            if column == "noaa_region_id":
                frame[column] = pd.to_numeric(
                    frame[column], errors="raise"
                ).astype("Int64")
            else:
                frame[column] = pd.to_datetime(
                    frame[column], utc=True, errors="raise"
                )

        if frame[keys].isna().any().any():
            raise ValueError(f"{source} contains missing matching keys.")

        if "noaa_region_id" in keys:
            if (frame["noaa_region_id"] <= 0).any():
                raise ValueError(f"{source} contains invalid region IDs.")

        frame["probability"] = pd.to_numeric(
            frame["probability"], errors="coerce"
        )

        # Remove exact duplicates, but do not hide conflicting forecasts.
        frame = frame.drop_duplicates()

        if frame.duplicated(keys, keep=False).any():
            raise ValueError(
                f"{source} has conflicting records for the same keys."
            )

        return frame

    hapi = prepare(hapi, "HAPI")
    archive = prepare(archive, "Archive")

    comparison = hapi.merge(
        archive,
        on=keys,
        how="outer",
        suffixes=("_hapi", "_archive"),
        indicator=True,
        validate="one_to_one",
    )

    overlap = comparison["_merge"] == "both"

    usable = overlap.copy()
    for column in ("probability_hapi", "probability_archive"):
        usable &= (
            comparison[column].notna()
            & comparison[column].between(0, 1)
        )

    comparison["absolute_difference"] = (
        comparison["probability_hapi"]
        - comparison["probability_archive"]
    ).abs()

    comparison["status"] = comparison["_merge"].map({
        "left_only": "hapi_only",
        "right_only": "archive_only",
        "both": "invalid_or_missing_probability",
    }).astype(str)

    matches = usable & np.isclose(
        comparison["probability_hapi"],
        comparison["probability_archive"],
        atol=tolerance,
        rtol=0,
        equal_nan=False,
    )

    comparison.loc[matches, "status"] = "match"
    comparison.loc[usable & ~matches, "status"] = "difference"

    compared_count = int(usable.sum())
    match_count = int(matches.sum())

    summary = {
        "hapi_records": len(hapi),
        "archive_records": len(archive),
        "overlapping_records": int(overlap.sum()),
        "hapi_only": int((comparison["_merge"] == "left_only").sum()),
        "archive_only": int((comparison["_merge"] == "right_only").sum()),
        "probability_matches": match_count,
        "probability_differences": int((usable & ~matches).sum()),
        "overlap_with_invalid_probability": int((overlap & ~usable).sum()),
        "agreement_percent": (
            100 * match_count / compared_count
            if compared_count else None
        ),
    }

    return summary, comparison.drop(columns="_merge")