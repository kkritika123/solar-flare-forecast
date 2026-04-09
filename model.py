# Evaluation: forecasts vs LMSAL events → TSS / HSS CSVs 
import glob
import os
from typing import List, Optional

import pandas as pd

from flare_eval_utils import (
    load_lmsal_events,
    load_model_forecasts,
    evaluate_yearly,
    evaluate_cumulative_running,
)
from flare_scoreboard.config import load_config

# We need to edit: None = score every subfolder of output/ that has *_full_disk.csv or *_region.csv.
# Or set an explicit list, e.g. ["NOAA_1", "SIDC_v2", "SPS_1"].
MODEL_NAMES: Optional[List[str]] = None

FORECAST_OUTPUT_DIR = "output"

LMSAL_CSV = "data/lmsal_events_2020_2025.csv"

PROBABILITY_CUTOFF = 0.5

# Skip empty score rows (n_forecasts=0); keep only rows with real forecast data.
OMIT_EMPTY_SCORE_ROWS = True

COMBINED_SCORES_CSV = os.path.join("evaluation_results", "all_models_scores.csv")
COMBINED_FULL_DISK_CSV = os.path.join(
    "evaluation_results", "all_models_full_disk_scores.csv"
)
COMBINED_REGION_CSV = os.path.join("evaluation_results", "all_models_region_scores.csv")
COMBINED_YEARLY_CSV = os.path.join("evaluation_results", "all_models_yearly_scores.csv")
COMBINED_CUMULATIVE_CSV = os.path.join(
    "evaluation_results", "all_models_cumulative_growing_scores.csv"
)


#  Column order for exported score tables 

def _reorder_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Put readable columns first (pool_description explains yearly vs cumulative)."""
    preferred = [
        "model_name",
        "evaluation_mode",
        "pool_description",
        "pool_n_calendar_years",
        "pool_year_span",
        "year",
        "forecast_type",
        "threshold",
        "n_forecasts",
        "TP",
        "FP",
        "FN",
        "TN",
        "TSS",
        "HSS",
    ]
    have = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in have]
    return df[have + rest].copy()


# Discover local forecast CSV folders
def discover_models_with_forecasts(out_dir: str) -> List[str]:
    """Subfolders of out_dir that contain at least one yearly forecast CSV."""
    if not os.path.isdir(out_dir):
        return []
    names: List[str] = []
    for entry in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, entry)
        if not os.path.isdir(sub) or entry.startswith("."):
            continue
        if glob.glob(os.path.join(sub, "*_full_disk.csv")) or glob.glob(
            os.path.join(sub, "*_region.csv")
        ):
            names.append(entry)
    return names


# Per-model TSS/HSS → evaluation_results/<MODEL>/ 
def evaluate_one_model(
    model_name: str,
    events_df: pd.DataFrame,
    *,
    tol_hours: float,
    eval_years: List[int],
    window_fill_hours: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    #Compute yearly and cumulative scores for one model using LMSAL event-window matching.
    model_dir = os.path.join(FORECAST_OUTPUT_DIR, model_name)
    results_dir = os.path.join("evaluation_results", model_name)
    y0, y1 = min(eval_years), max(eval_years)

    print(f"\n========== Model: {model_name} ==========")
    print(f"Loading forecasts from: {model_dir}")

    try:
        forecasts_df = load_model_forecasts(model_dir, years=eval_years)
    except FileNotFoundError:
        print(f"  [SKIP] No yearly CSVs in {model_dir}")
        return None

    print("  Forecast rows loaded:", len(forecasts_df))

    # Step 1 — individual (year-wise): one block of metrics per calendar year
    print("  Step 1: yearly (per-year TSS, HSS)...")
    yearly = evaluate_yearly(
        forecasts_df=forecasts_df,
        events_df=events_df,
        probability_cutoff=PROBABILITY_CUTOFF,
        tol_hours=tol_hours,
        omit_empty=OMIT_EMPTY_SCORE_ROWS,
        years_list=eval_years,
        window_fill_hours=window_fill_hours,
    )
    yearly.insert(0, "model_name", model_name)

    # Step 2 — growing cumulative: through y0, then y0..y0+1, … through y1
    print(f"  Step 2: cumulative running ({y0} … {y1})...")
    cumulative = evaluate_cumulative_running(
        forecasts_df=forecasts_df,
        events_df=events_df,
        probability_cutoff=PROBABILITY_CUTOFF,
        tol_hours=tol_hours,
        omit_empty=OMIT_EMPTY_SCORE_ROWS,
        year_start=y0,
        year_end=y1,
        window_fill_hours=window_fill_hours,
    )
    cumulative.insert(0, "model_name", model_name)

    scores = pd.concat([yearly, cumulative], ignore_index=True)

    if scores.empty or "forecast_type" not in scores.columns:
        print("  [WARN] No score rows (no data after omitting empty year/threshold cells).")
        return None

    scores = _reorder_score_columns(scores)

    os.makedirs(results_dir, exist_ok=True)

    fd = scores[scores["forecast_type"] == "full_disk"].copy()
    rg = scores[scores["forecast_type"] == "region"].copy()

    yearly_only = scores[scores["evaluation_mode"] == "yearly"].copy()
    cumulative_only = scores[scores["evaluation_mode"] == "cumulative_running"].copy()

    out_fd = os.path.join(results_dir, f"{model_name}_full_disk_scores.csv")
    out_rg = os.path.join(results_dir, f"{model_name}_region_scores.csv")
    out_all = os.path.join(results_dir, f"{model_name}_scores.csv")
    out_yearly = os.path.join(results_dir, f"{model_name}_yearly_scores.csv")
    out_cum = os.path.join(results_dir, f"{model_name}_cumulative_growing_scores.csv")

    fd.to_csv(out_fd, index=False)
    scores.to_csv(out_all, index=False)
    yearly_only.to_csv(out_yearly, index=False)
    cumulative_only.to_csv(out_cum, index=False)

    print(f"  Saved (full disk): {out_fd}")
    if len(rg) > 0:
        rg.to_csv(out_rg, index=False)
        print(f"  Saved (region):    {out_rg}")
    else:
        if os.path.isfile(out_rg):
            try:
                os.remove(out_rg)
            except OSError:
                pass
        print(
            "  [SKIP] No region forecast rows for this model — "
            "region scores CSV not written (see Methods: not all models submit region forecasts)."
        )
    print(f"  Saved (combined):  {out_all}")
    print(f"  Saved (yearly only):              {out_yearly}")
    print(f"  Saved (cumulative growing only):  {out_cum}")

    for label, part in (("full_disk", fd), ("region", rg)):
        if len(part):
            print(f"\n  --- {label} ---")
            print(
                part.sort_values(
                    ["evaluation_mode", "year", "threshold"]
                ).to_string(index=False)
            )

    return scores


# Entry: all models -> combined CSVs in evaluation_results/
def main():
    os.makedirs("evaluation_results", exist_ok=True)

    cfg = load_config()
    tol_hours = cfg["event_window_match_tolerance_hours"]
    eval_years = sorted(int(y) for y in cfg["years_set"])
    wf = cfg.get("forecast_window_fill_hours")
    print(
        "Event vs forecast window matching: LMSAL event_start_utc must fall inside "
        f"each row's window_begin..window_end (UTC), +/-{tol_hours} h padding."
    )
    if tol_hours == 0:
        print("  (0 h = strict - no extra slack; comparable across models' published windows.)")
    if wf:
        print(
            f"Rows with no window in file: using issue_time_UTC -> issue_time + {wf:g} h "
            "as validity interval before +/- tolerance."
        )

    print(f"Loading LMSAL events from: {LMSAL_CSV}")
    events_df = load_lmsal_events(LMSAL_CSV)
    print("LMSAL rows loaded:", len(events_df))

    # Either use explicit model list or auto-discover from output/ folders.
    if MODEL_NAMES is not None:
        model_names = list(MODEL_NAMES)
        print(f"Using MODEL_NAMES list ({len(model_names)} model(s)).")
    else:
        model_names = discover_models_with_forecasts(FORECAST_OUTPUT_DIR)
        print(
            f"Auto-discovered {len(model_names)} model folder(s) under "
            f"{FORECAST_OUTPUT_DIR!r} with *_full_disk.csv or *_region.csv."
        )
    if not model_names:
        print("No models to evaluate. Build forecasts with main.py or set MODEL_NAMES.")
        return
    for m in model_names:
        print(" -", m)

    all_parts: List[pd.DataFrame] = []

    for name in model_names:
        part = evaluate_one_model(
            name,
            events_df,
            tol_hours=tol_hours,
            eval_years=eval_years,
            window_fill_hours=wf,
        )
        if part is not None:
            all_parts.append(part)

    if all_parts:
        # Merge per-model tables into combined exports for plotting/reporting.
        combined = _reorder_score_columns(pd.concat(all_parts, ignore_index=True))
        combined.to_csv(COMBINED_SCORES_CSV, index=False)
        fd_all = combined[combined["forecast_type"] == "full_disk"].copy()
        rg_all = combined[combined["forecast_type"] == "region"].copy()
        fd_all.to_csv(COMBINED_FULL_DISK_CSV, index=False)
        y_all = combined[combined["evaluation_mode"] == "yearly"].copy()
        c_all = combined[combined["evaluation_mode"] == "cumulative_running"].copy()
        y_all.to_csv(COMBINED_YEARLY_CSV, index=False)
        c_all.to_csv(COMBINED_CUMULATIVE_CSV, index=False)
        print(f"\nCombined (all types): {COMBINED_SCORES_CSV}")
        print(f"Combined full disk only: {COMBINED_FULL_DISK_CSV}")
        if len(rg_all) > 0:
            rg_all.to_csv(COMBINED_REGION_CSV, index=False)
            print(f"Combined region only:    {COMBINED_REGION_CSV}")
        else:
            if os.path.isfile(COMBINED_REGION_CSV):
                try:
                    os.remove(COMBINED_REGION_CSV)
                except OSError:
                    pass
            print(
                "Combined region only:    [not written — no region rows in any evaluated model]"
            )
        print(f"Combined yearly only:              {COMBINED_YEARLY_CSV}")
        print(f"Combined cumulative growing only: {COMBINED_CUMULATIVE_CSV}")
    else:
        print("\nNo models were evaluated (check output/ folders and MODEL_NAMES).")


if __name__ == "__main__":
    main()
