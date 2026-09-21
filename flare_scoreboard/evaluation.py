"""Matching and scoring helpers used by model.py.

For each forecast row, y_pred = (probability >= cutoff) and y_true = 1 if any
qualifying LMSAL flare starts inside the forecast window. TSS/HSS come from
the TP/FP/FN/TN counts.
"""

import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ORDER = {"C": 1, "M": 2, "X": 3}


def to_dt(x):
    """Convert anything time-like to a UTC pandas timestamp (NaT on failure)."""
    return pd.to_datetime(x, errors="coerce", utc=True)


def normalize_region_id(x) -> str:
    """Clean NOAA active-region ids so forecasts and LMSAL agree (e.g. 4274.0 -> 4274, 12774 -> 2774)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".", 1)[0]
    if re.fullmatch(r"1\d{4}", s):
        s = str(int(s) - 10000)
    return s


def normalize_threshold(x: str) -> str:
    """Return 'C', 'M', or 'X' if the string starts with one; else empty."""
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if s and s[0] in {"C", "M", "X"}:
        return s[0]
    return ""


def parse_flare_letter(x: str) -> str:
    """Return the GOES class letter (C, M, or X) from a class string like 'M5.1'."""
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


def _pool_span_label(sorted_years: List[int]) -> str:
    """Format a year span, e.g. [2024, 2025] -> '2024-2025'."""
    if not sorted_years:
        return ""
    if len(sorted_years) == 1:
        return str(sorted_years[0])
    return f"{sorted_years[0]}-{sorted_years[-1]}"


def _events_meet_threshold_mask(events_df: pd.DataFrame, threshold: str) -> pd.Series:
    letters = events_df["flare_letter"].astype(str).str[0].str.upper()
    ev_order = letters.map(ORDER).fillna(0).astype(int)
    thr = ORDER.get(threshold, 999)
    return ev_order >= thr


def _dt_series_to_i64_ns(s: pd.Series) -> np.ndarray:
    """UTC datetimes as int64 nanoseconds for fast numpy searchsorted."""
    return pd.to_datetime(s, utc=True).astype("int64").to_numpy()


def fill_missing_windows_from_issue(
    df: pd.DataFrame, hours: Optional[float]
) -> pd.DataFrame:
    """If both window times are missing, use [issue_time, issue_time + hours]."""
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
    return sorted(ys) if ys else sorted(set(fallback_years))


def _y_true_vectorized(
    df: pd.DataFrame,
    e_year: pd.DataFrame,
    forecast_type: str,
    threshold: str,
    tol_hours: float,
) -> np.ndarray:
    """y_true = 1 where a qualifying flare starts in the window (region forecasts also need matching AR id)."""
    n = len(df)
    out = np.zeros(n, dtype=np.int8)
    if n == 0 or e_year.empty:
        return out

    e_cand = e_year.loc[_events_meet_threshold_mask(e_year, threshold)]
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
            sub_e = e_nonempty[e_nonempty["region_id"].astype(str).str.strip() == Rs]
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
    """Alias for normalize_region_id kept for readability."""
    return normalize_region_id(x)


def calc_scores(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Return TP, FP, FN, TN, TSS and HSS from the four contingency counts."""
    pod_den = tp + fn
    pofd_den = fp + tn

    pod = tp / pod_den if pod_den else np.nan
    pofd = fp / pofd_den if pofd_den else np.nan
    tss = pod - pofd if (not np.isnan(pod) and not np.isnan(pofd)) else np.nan

    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fn * fp) / hss_den) if hss_den else np.nan

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "TSS": tss, "HSS": hss}


def load_lmsal_events(csv_path: str) -> pd.DataFrame:
    """Load the LMSAL events CSV and normalize the columns used for matching."""
    df = pd.read_csv(csv_path)
    df["event_start_utc"] = pd.to_datetime(df["event_start_utc"], utc=True)
    df["flare_letter"] = df["flare_class"].astype(str).str[0]
    df["year"] = df["event_start_utc"].dt.year
    df["region_id"] = df["region_id"].apply(normalize_region_id)
    return df


def load_model_forecasts(
    model_dir: str,
    years: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Read every <YEAR>_full_disk.csv and <YEAR>_region.csv under a model folder."""
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
            d["issue_time_utc"] = to_dt(d["issue_time_utc"]) if "issue_time_utc" in d.columns else pd.NaT

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


def build_binary_rows(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    forecast_type: str,
    threshold: str,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Filter by forecast type and threshold, fill missing windows, add y_pred and y_true."""
    df = forecasts_df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df[df["forecast_type"] == forecast_type].copy()
    df = df[df["flare_threshold"] == threshold].copy()

    df = fill_missing_windows_from_issue(df, window_fill_hours)
    df["y_pred"] = (df["probability"] >= probability_cutoff).astype(int)
    df["y_true"] = _y_true_vectorized(df, events_df, forecast_type, threshold, tol_hours).astype(int)
    return df


def scores_from_binary_df(df: pd.DataFrame) -> Dict[str, float]:
    tp = int(((df["y_pred"] == 1) & (df["y_true"] == 1)).sum())
    fp = int(((df["y_pred"] == 1) & (df["y_true"] == 0)).sum())
    fn = int(((df["y_pred"] == 0) & (df["y_true"] == 1)).sum())
    tn = int(((df["y_pred"] == 0) & (df["y_true"] == 0)).sum())
    return calc_scores(tp, fp, fn, tn)


def _yearly_pool_description(year: int) -> str:
    return (
        f"Per-year only: uses forecasts and events in calendar year {year} "
        "(not mixed with other years)."
    )


def _cumulative_pool_description(end_y: int, n_years: int, span: str) -> str:
    return (
        f"Growing cumulative through end year {end_y}: pools every forecast/event "
        f"row with calendar year <= {end_y}. With your data this is "
        f"{n_years} distinct year(s): {span}."
    )


def evaluate_yearly(
    forecasts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    probability_cutoff: float = 0.5,
    tol_hours: float = 0.0,
    omit_empty: bool = True,
    years_list: Optional[List[int]] = None,
    window_fill_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Score per calendar year: one 2x2 per (year, forecast_type, threshold)."""
    rows: List[dict] = []
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

                base = {
                    "evaluation_mode": "yearly",
                    "pool_description": _yearly_pool_description(year),
                    "pool_n_calendar_years": 1,
                    "pool_year_span": str(year),
                    "year": year,
                    "forecast_type": ftype,
                    "threshold": thr,
                }

                if binary_df.empty:
                    if omit_empty:
                        continue
                    rows.append({**base, "n_forecasts": 0, "TP": 0, "FP": 0, "FN": 0,
                                 "TN": 0, "TSS": np.nan, "HSS": np.nan})
                    continue

                s = scores_from_binary_df(binary_df)
                rows.append({**base, "n_forecasts": len(binary_df), **s})

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
    """Growing-window scores: for each end year Y, score everything with year <= Y."""
    rows: List[dict] = []

    for end_y in range(year_start, year_end + 1):
        f_sub = forecasts_df[forecasts_df["year"] <= end_y].copy()
        e_sub = events_df[events_df["year"] <= end_y].copy()
        fy = sorted({int(y) for y in f_sub["year"].dropna().unique()})
        span = _pool_span_label(fy)
        n_years = len(fy)

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

                base = {
                    "evaluation_mode": "cumulative_running",
                    "pool_description": _cumulative_pool_description(end_y, n_years, span),
                    "pool_n_calendar_years": n_years,
                    "pool_year_span": span,
                    "year": end_y,
                    "forecast_type": ftype,
                    "threshold": thr,
                }

                if binary_df.empty:
                    if omit_empty:
                        continue
                    rows.append({**base, "n_forecasts": 0, "TP": 0, "FP": 0, "FN": 0,
                                 "TN": 0, "TSS": np.nan, "HSS": np.nan})
                    continue

                s = scores_from_binary_df(binary_df)
                rows.append({**base, "n_forecasts": len(binary_df), **s})

    return pd.DataFrame(rows)
