# ── Write normalized yearly CSVs per model ───────────────────────────────────
import csv
import os
from typing import Dict, List, Optional, Set

from flare_scoreboard.constants import FIELDNAMES
from flare_scoreboard.parse_core import year_from_issue_time


def write_csv(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.isdir(out_path):
        raise IsADirectoryError(f"Expected file path, but found folder: {out_path}")

    # Write to a temp file first, then replace atomically.
    tmp_path = out_path + ".tmp"

    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        os.replace(tmp_path, out_path)

    except PermissionError:
        print(f"[WRITE FAILED] Close this file if it is open: {out_path}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _remove_csv_if_exists(path: str) -> None:
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def write_model_year_csvs_split(
    model_rows: List[Dict],
    model_name: str,
    out_dir: str,
    years_set: Optional[Set[int]] = None,
):
    """
    Split rows into yearly CSVs by issue_time_utc year.
    If years_set is given, only those calendar years are written (matches config "years").
    """
    # buckets[year]["full_disk"|"region"] -> list of normalized rows
    buckets: Dict[int, Dict[str, List[Dict]]] = {}

    for r in model_rows:
        y = year_from_issue_time(r.get("issue_time_utc", ""))
        if y is None:
            continue
        if years_set is not None and y not in years_set:
            continue

        row_copy = dict(r)
        row_copy["year"] = y

        ftype = row_copy.get("forecast_type", "full_disk")
        if ftype not in ("full_disk", "region"):
            ftype = "full_disk"

        buckets.setdefault(y, {"full_disk": [], "region": []})
        buckets[y][ftype].append(row_copy)

    if model_rows and not buckets:
        print(
            "  [WARN] No rows written: issue_time_utc missing or year outside config "
            f"({len(model_rows)} parsed rows dropped). Check .txt filename/URL patterns."
        )

    model_out = os.path.join(out_dir, model_name)
    os.makedirs(model_out, exist_ok=True)

    # Write one file per year and forecast stream.
    for y in sorted(buckets.keys()):
        out_full = os.path.join(model_out, f"{y}_full_disk.csv")
        out_region = os.path.join(model_out, f"{y}_region.csv")
        fd_rows = buckets[y]["full_disk"]
        rg_rows = buckets[y]["region"]

        if fd_rows:
            print(f"  Writing: {out_full}")
            write_csv(fd_rows, out_full)
        else:
            _remove_csv_if_exists(out_full)

        if rg_rows:
            print(f"  Writing: {out_region}")
            write_csv(rg_rows, out_region)
        else:
            _remove_csv_if_exists(out_region)
