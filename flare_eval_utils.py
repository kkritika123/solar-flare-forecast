# Forecast verification vs LMSAL using event-window matching; outputs TP/FP/FN/TN, TSS, and HSS.

import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ORDER = {"C": 1, "M": 2, "X": 3}


def to_dt(x):
    return pd.to_datetime(x, errors="coerce", utc=True)


def normalize_region_id(x) -> str:
    """
    Normalize NOAA AR IDs for matching:
    - strip float-like suffix from CSV inference (4274.0 -> 4274)
    - align 5-digit NOAA form with LMSAL 4-digit form (12774 -> 2774)
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    # Handle CSV numeric inference: 4274 -> 4274.0 when read as float.
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    # Some model feeds use 1xxxx while LMSAL events use the de-offset form.
    if re.fullmatch(r"1\d{4}", s):
        s = str(int(s) - 10000)
    return s


def normalize_threshold(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if not s:
        return ""
    if s[0] in {"C", "M", "X"}:
        return s[0]
    return ""


def _pool_span_label(sorted_years: List[int]) -> str:
    """e.g. [2024] -> '2024'; [2024, 2025] -> '2024-2025' for clear cumulative tables."""
    if not sorted_years:
        return ""
    if len(sorted_years) == 1:
        return str(sorted_years[0])
    return f"{sorted_years[0]}-{sorted_years[-1]}"


def parse_flare_letter(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if not s:
        return ""
    m = re.search(r"([CMX])\s*\d+(\.\d+)?", s)
    if m:
        return m.group(1)
    return s[0] if s[0] in {"C", "M", "X"} else ""


def threshold_met(event_letter: str, threshold: str) -> bool:
    return ORDER.get(event_letter, 0) >= ORDER.get(threshold, 999)


def _events_meet_threshold_mask(events_df: pd.DataFrame, threshold: str) -> pd.Series:
    letters = events_df["flare_letter"].astype(str).str[0].str.upper()
    ev_order = letters.map(ORDER).fillna(0).astype(int)
    thr = ORDER.get(threshold, 999)
    return ev_order >= thr


def _dt_series_to_i64_ns(s: pd.Series) -> np.ndarray:
    """UTC-ish datetimes as int64 ns for numpy searchsorted."""
    return pd.to_datetime(s, utc=True).astype("int64").to_numpy()


def fill_missing_windows_from_issue(
    df: pd.DataFrame, hours: Optional[float]
) -> pd.DataFrame:
    # If window bounds are missing, infer them as issue_time to issue_time + hours.
    if hours is None or float(hours) <= 0:
        return df
    out = df.copy()
    if "issue_time_utc" not in out.columns:
        return out
    wb = pd.to_datetime(out["window_begin_utc"], errors="coerce", utc=True)
    we = pd.to_datetime(out["window_end_utc"], errors="coerce", utc=True)
    it = pd.to_datetime(out["issue_time_utc"], errors="coerce", utc=True)
    both_missing = wb.isna() & we.isna() & it.notna()
    if not both_missing.any():
        return out
    delta = pd.Timedelta(hours=float(hours))
    wb = wb.copy()
    we = we.copy()
    wb.loc[both_missing] = it.loc[both_missing]
    we.loc[both_missing] = it.loc[both_missing] + delta
    out["window_begin_utc"] = wb
    out["window_end_utc"] = we
    return out


def evaluation_calendar_years(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fallback_years: Iterable[int],
) -> List[int]:
    ys = set()
    if "year" in forecasts_df.columns:
        ys |= set(pd.to_numeric(forecasts_df["year"], errors="coerce").dropna().astype(int))
    if "year" in events_df.columns:
        ys |= set(pd.to_numeric(events_df["year"], errors="coerce").dropna().astype(int))
    if not ys:
        return sorted(set(fallback_years))
    return sorted(ys)


def _y_true_vectorized(
    df: pd.DataFrame,
    e_year: pd.DataFrame,
    forecast_type: str,
    threshold: str,
    tol_hours: float,
) -> np.ndarray:
   
   # Fast vectorized event-window matching for y_true, with class and region filters.
    n = len(df)
    out = np.zeros(n, dtype=np.int8)
    if n == 0 or e_year.empty:
        return out

    mask_thr = _events_meet_threshold_mask(e_year, threshold)
    e_cand = e_year.loc[mask_thr]
    if e_cand.empty:
        return out

    tol = pd.Timedelta(hours=tol_hours)
    ftype = str(forecast_type).lower()

    work = df.reset_index(drop=True)
    wb = work["window_begin_utc"]
    we = work["window_end_utc"]
    bad_win = wb.isna() | we.isna()

    if ftype == "full_disk":
        es = np.sort(_dt_series_to_i64_ns(e_cand["event_start_utc"].dropna()))
        if es.size == 0:
            return out
        wbv = _dt_series_to_i64_ns(wb - tol)
        wev = _dt_series_to_i64_ns(we + tol)
        lo = np.searchsorted(es, wbv, side="left")
        hi = np.searchsorted(es, wev, side="right")
        m = (hi > lo) & (~bad_win.to_numpy())
        return m.astype(np.int8)

    if ftype == "region":
        reg_f = work["region_id"].astype(str).str.strip()
        empty_reg = bad_win.to_numpy() | reg_f.isin(["", "nan", "None"]).to_numpy()

        e_nonempty = e_cand[e_cand["region_id"].astype(str).str.strip() != ""]
        if e_nonempty.empty:
            return out

        for R, g in work.groupby(reg_f, sort=False):
            Rs = str(R).strip()
            if Rs in ("", "nan", "None"):
                continue
            idx = g.index.to_numpy(dtype=np.intp)
            sub_e = e_nonempty[
                e_nonempty["region_id"].astype(str).str.strip() == Rs
            ]
            if sub_e.empty:
                continue
            es = np.sort(_dt_series_to_i64_ns(sub_e["event_start_utc"].dropna()))
            if es.size == 0:
                continue
            gw = g["window_begin_utc"].isna() | g["window_end_utc"].isna()
            wbv = _dt_series_to_i64_ns(g["window_begin_utc"] - tol)
            wev = _dt_series_to_i64_ns(g["window_end_utc"] + tol)
            lo = np.searchsorted(es, wbv, side="left")
            hi = np.searchsorted(es, wev, side="right")
            m = (hi > lo) & (~gw.to_numpy())
            out[idx] = m.astype(np.int8)

        out[empty_reg] = 0
        return out

    return out


def safe_region_id(x) -> str:
    return normalize_region_id(x)


def calc_scores(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    
    pod_den = tp + fn
    pofd_den = fp + tn

    pod = tp / pod_den if pod_den else np.nan
    pofd = fp / pofd_den if pofd_den else np.nan
    tss = pod - pofd if (not np.isnan(pod) and not np.isnan(pofd)) else np.nan

    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fn * fp) / hss_den) if hss_den else np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TSS": tss,
        "HSS": hss,
    }


def load_lmsal_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # convert time column
    df["event_start_utc"] = pd.to_datetime(df["event_start_utc"], utc=True)

    # create flare letter column (C/M/X)
    df["flare_letter"] = df["flare_class"].astype(str).str[0]

    # create year column
    df["year"] = df["event_start_utc"].dt.year

    # Normalize region IDs so they match forecast IDs ("4274", not "4274.0").
    df["region_id"] = df["region_id"].apply(normalize_region_id)

    return df


def load_model_forecasts(
    model_dir: str,
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    year_list = sorted(set(int(y) for y in years)) if years is not None else list(range(2020, 2026))
    ymin, ymax = min(year_list), max(year_list)

    for year in year_list:
        for ftype in ("full_disk", "region"):
            path = os.path.join(model_dir, f"{year}_{ftype}.csv")
            if not os.path.exists(path):
                continue

            d = pd.read_csv(path)
            d.columns = [c.strip().lower() for c in d.columns]

            required = ["window_begin_utc", "window_end_utc", "flare_threshold", "probability"]
            missing = [c for c in required if c not in d.columns]
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")

            d["window_begin_utc"] = to_dt(d["window_begin_utc"])
            d["window_end_utc"] = to_dt(d["window_end_utc"])

            if "issue_time_utc" in d.columns:
                d["issue_time_utc"] = to_dt(d["issue_time_utc"])
            else:
                d["issue_time_utc"] = pd.NaT

            d["forecast_type"] = ftype
            if "region_id" not in d.columns:
                d["region_id"] = ""
            d["region_id"] = d["region_id"].apply(safe_region_id)

            d["flare_threshold"] = d["flare_threshold"].apply(normalize_threshold)
            d["probability"] = pd.to_numeric(d["probability"], errors="coerce")
            d["year"] = year

            d = d[d["flare_threshold"].isin(["C", "M", "X"])].copy()
            dfs.append(d)

    if not dfs:
        raise FileNotFoundError(f"No model yearly CSVs found in {model_dir}")

    out = pd.concat(dfs, ignore_index=True)
    out = out[out["year"].between(ymin, ymax, inclusive="both")].copy()
    return out


def start_time_matches(event_start, win_begin, win_end, tol_hours: float = 0.0) -> bool:
    """
    Return True if event_start_utc is inside the forecast window.
    Matching uses inclusive bounds:
    (win_begin - tol) <= event_start <= (win_end + tol).
    tol_hours=0 means strict window-only matching.
    """
    if pd.isna(event_start) or pd.isna(win_begin) or pd.isna(win_end):
        return False

    tol = pd.Timedelta(hours=tol_hours)
    return (win_begin - tol) <= event_start <= (win_end + tol)


def row_has_matching_event(forecast_row: pd.Series, events_df: pd.DataFrame, tol_hours: float = 0.0) -> bool:
    threshold = forecast_row["flare_threshold"]
    ftype = str(forecast_row["forecast_type"]).lower()
    model_region = safe_region_id(forecast_row.get("region_id", ""))

    candidates = events_df.copy()
    candidates = candidates[candidates["year"] == forecast_row["year"]]
    candidates = candidates[candidates["flare_letter"].apply(lambda x: threshold_met(x, threshold))]

    candidates = candidates[
        candidates["event_start_utc"].apply(
            lambda dt: start_time_matches(
                dt,
                forecast_row["window_begin_utc"],
                forecast_row["window_end_utc"],
                tol_hours=tol_hours,
            )
        )
    ]

    if candidates.empty:
        return False

    if ftype == "full_disk":
        return True

    if ftype == "region":
        if model_region:
            both_with_region = candidates[candidates["region_id"].astype(str).str.strip() != ""]
            if both_with_region.empty:
                return False
            return (both_with_region["region_id"].astype(str).str.strip() == model_region).any()
        return False

    return False


def build_binary_rows(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    forecast_type: str,
    threshold: str,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:

    df = forecasts_df.copy()

    # remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df = df[df["forecast_type"] == forecast_type].copy()
    df = df[df["flare_threshold"] == threshold].copy()

    df = fill_missing_windows_from_issue(df, window_fill_hours)

    # prediction label
    df["y_pred"] = (df["probability"] >= probability_cutoff).astype(int)

    y_true = _y_true_vectorized(df, events_df, forecast_type, threshold, tol_hours)
    df["y_true"] = y_true.astype(int)

    return df

def scores_from_binary_df(df: pd.DataFrame) -> Dict[str, float]:
    tp = int(((df["y_pred"] == 1) & (df["y_true"] == 1)).sum())
    fp = int(((df["y_pred"] == 1) & (df["y_true"] == 0)).sum())
    fn = int(((df["y_pred"] == 0) & (df["y_true"] == 1)).sum())
    tn = int(((df["y_pred"] == 0) & (df["y_true"] == 0)).sum())
    return calc_scores(tp, fp, fn, tn)


def evaluate_yearly(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    omit_empty: bool = True,
    years_list: Optional[List[int]] = None,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:
    
    #omit_empty drops n_forecasts=0 rows; years_list controls which calendar years are scored.
    rows = []

    years_loop = (
        years_list
        if years_list is not None
        else evaluation_calendar_years(forecasts_df, events_df, range(2020, 2026))
    )

    for year in years_loop:
        f_year = forecasts_df[forecasts_df["year"] == year].copy()
        e_year = events_df[events_df["year"] == year].copy()

        for ftype in ("full_disk", "region"):
            for thr in ("C", "M", "X"):
                binary_df = build_binary_rows(
                    forecasts_df=f_year,
                    events_df=e_year,
                    forecast_type=ftype,
                    threshold=thr,
                    probability_cutoff=probability_cutoff,
                    tol_hours=tol_hours,
                    window_fill_hours=window_fill_hours,
                )

                if binary_df.empty:
                    if omit_empty:
                        continue
                    rows.append({
                        "evaluation_mode": "yearly",
                        "pool_description": (
                            f"Per-year only: uses forecasts and events in calendar year {year} "
                            "(not mixed with other years)."
                        ),
                        "pool_n_calendar_years": 1,
                        "pool_year_span": str(year),
                        "year": year,
                        "forecast_type": ftype,
                        "threshold": thr,
                        "n_forecasts": 0,
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "TN": 0,
                        "TSS": np.nan,
                        "HSS": np.nan,
                    })
                    continue

                s = scores_from_binary_df(binary_df)
                rows.append({
                    "evaluation_mode": "yearly",
                    "pool_description": (
                        f"Per-year only: uses forecasts and events in calendar year {year} "
                        "(not mixed with other years)."
                    ),
                    "pool_n_calendar_years": 1,
                    "pool_year_span": str(year),
                    "year": year,
                    "forecast_type": ftype,
                    "threshold": thr,
                    "n_forecasts": len(binary_df),
                    **s,
                })

    return pd.DataFrame(rows)


def evaluate_cumulative_running(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    year_start: int = 2020,
    year_end: int = 2025,
    omit_empty: bool = True,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:
    """
    Growing-window cumulative metrics: pool data for calendar years year_start..end_y
    for each end_y in [year_start, year_end]. E.g. end_y=2021 uses 2020+2021 only.
    """
    rows = []

    for end_y in range(year_start, year_end + 1):
        f_sub = forecasts_df[forecasts_df["year"] <= end_y].copy()
        e_sub = events_df[events_df["year"] <= end_y].copy()

        for ftype in ("full_disk", "region"):
            for thr in ("C", "M", "X"):
                binary_df = build_binary_rows(
                    forecasts_df=f_sub,
                    events_df=e_sub,
                    forecast_type=ftype,
                    threshold=thr,
                    probability_cutoff=probability_cutoff,
                    tol_hours=tol_hours,
                    window_fill_hours=window_fill_hours,
                )

                if binary_df.empty:
                    if omit_empty:
                        continue
                    fy = sorted({int(y) for y in f_sub["year"].dropna().unique()})
                    span_e = _pool_span_label(fy)
                    ny_e = len(fy)
                    rows.append({
                        "evaluation_mode": "cumulative_running",
                        "pool_description": (
                            f"Growing cumulative through end year {end_y}: pools every forecast/event "
                            f"row with calendar year ≤ {end_y}. "
                            f"With your data this is {ny_e} distinct year(s): {span_e}."
                        ),
                        "pool_n_calendar_years": ny_e,
                        "pool_year_span": span_e,
                        "year": end_y,
                        "forecast_type": ftype,
                        "threshold": thr,
                        "n_forecasts": 0,
                        "TP": 0,
                        "FP": 0,
                        "FN": 0,
                        "TN": 0,
                        "TSS": np.nan,
                        "HSS": np.nan,
                    })
                    continue

                s = scores_from_binary_df(binary_df)
                fy = sorted({int(y) for y in f_sub["year"].dropna().unique()})
                span = _pool_span_label(fy)
                ny = len(fy)
                rows.append({
                    "evaluation_mode": "cumulative_running",
                    "pool_description": (
                        f"Growing cumulative through end year {end_y}: pools every forecast/event "
                        f"row with calendar year ≤ {end_y}. "
                        f"With your data this is {ny} distinct year(s): {span}."
                    ),
                    "pool_n_calendar_years": ny,
                    "pool_year_span": span,
                    "year": end_y,
                    "forecast_type": ftype,
                    "threshold": thr,
                    "n_forecasts": len(binary_df),
                    **s,
                })

    return pd.DataFrame(rows)


def evaluate_cumulative(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    omit_empty: bool = True,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    fy_all = sorted({int(y) for y in forecasts_df["year"].dropna().unique()})
    span_all = _pool_span_label(fy_all)
    ny_all = len(fy_all)

    for ftype in ("full_disk", "region"):
        for thr in ("C", "M", "X"):
            binary_df = build_binary_rows(
                forecasts_df=forecasts_df,
                events_df=events_df,
                forecast_type=ftype,
                threshold=thr,
                probability_cutoff=probability_cutoff,
                tol_hours=tol_hours,
                window_fill_hours=window_fill_hours,
            )

            if binary_df.empty:
                if omit_empty:
                    continue
                rows.append({
                    "evaluation_mode": "cumulative_all",
                    "pool_description": (
                        "Full cumulative: single 2x2 table using every loaded forecast and event row."
                    ),
                    "pool_n_calendar_years": ny_all,
                    "pool_year_span": span_all,
                    "year": "cumulative",
                    "forecast_type": ftype,
                    "threshold": thr,
                    "n_forecasts": 0,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "TN": 0,
                    "TSS": np.nan,
                    "HSS": np.nan,
                })
                continue

            s = scores_from_binary_df(binary_df)
            rows.append({
                "evaluation_mode": "cumulative_all",
                "pool_description": (
                    "Full cumulative: single 2x2 table using every loaded forecast and event row."
                ),
                "pool_n_calendar_years": ny_all,
                "pool_year_span": span_all,
                "year": "cumulative",
                "forecast_type": ftype,
                "threshold": thr,
                "n_forecasts": len(binary_df),
                **s,
            })

    return pd.DataFrame(rows)


def export_binary_labels(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: str,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    years_list: Optional[List[int]] = None,
    window_fill_hours: Optional[float] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    years_loop = (
        years_list
        if years_list is not None
        else evaluation_calendar_years(forecasts_df, events_df, range(2020, 2026))
    )

    for year in years_loop:
        f_year = forecasts_df[forecasts_df["year"] == year].copy()
        e_year = events_df[events_df["year"] == year].copy()

        for ftype in ("full_disk", "region"):
            for thr in ("C", "M", "X"):
                df = build_binary_rows(
                    forecasts_df=f_year,
                    events_df=e_year,
                    forecast_type=ftype,
                    threshold=thr,
                    probability_cutoff=probability_cutoff,
                    tol_hours=tol_hours,
                    window_fill_hours=window_fill_hours,
                )
                if not df.empty:
                    out_path = os.path.join(out_dir, f"{year}_{ftype}_{thr}_labels.csv")
                    df.to_csv(out_path, index=False)