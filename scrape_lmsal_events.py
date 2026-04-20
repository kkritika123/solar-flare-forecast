"""Scrape LMSAL 'latest events' pages and build data/lmsal_events_YYYY_YYYY.csv.

Used as the ground truth for model.py. Writes one combined CSV per date range.
"""
import argparse
import os
import re
from datetime import datetime, timedelta
from io import StringIO
from urllib.parse import urljoin
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

ARCHIVE_URL = "https://www.lmsal.com/solarsoft/latest_events_archive.html"


def safe_get(url: str, timeout: int = 30) -> Tuple[Optional[str], int]:
    headers = {"User-Agent": "Mozilla/5.0 (SSW-scraper/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        status = r.status_code
        if status == 404:
            return None, status
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text, status
    except requests.RequestException:
        return None, -1


def get_date_from_last_events_url(url: str):
    m = re.search(r"last_events_(\d{8})_\d{4}", url)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def parse_position(pos_text: str):
    if pos_text is None:
        return None, None, None

    s = str(pos_text).strip()
    if s in {"", "-", "nan"}:
        return None, None, None

    m = re.search(r"([NS]\d+)([EW]\d+)\s*\(\s*(\d+)\s*\)", s)
    if m:
        return m.group(1), m.group(2), m.group(3)

    m2 = re.search(r"([NS]\d+)([EW]\d+)", s)
    if m2:
        return m2.group(1), m2.group(2), None

    return None, None, None


def parse_datetime_or_none(date_str: str, time_str: str):
    if time_str is None:
        return None
    t = str(time_str).strip()
    if t in {"", "-", "nan"}:
        return None
    return datetime.strptime(f"{date_str} {t}", "%Y/%m/%d %H:%M:%S")


def find_event_table_from_html(html: str):
    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            return None

    def norm_cols(df: pd.DataFrame):
        return {str(c).strip().lower() for c in df.columns}

    for t in tables:
        cols = norm_cols(t)
        if "ename" in cols and "start" in cols:
            if any(("goes" in c and "class" in c) for c in cols):
                return t.copy()

    return None


def scrape_day_url(url: str):
    html, status = safe_get(url)
    if html is None:
        return [], True

    event_df = find_event_table_from_html(html)
    if event_df is None or event_df.empty:
        return [], False

    cols_lower = {str(c).strip().lower(): c for c in event_df.columns}

    def col(name: str):
        return cols_lower.get(name.lower())

    c_ename = col("EName")
    c_start = col("Start")
    c_eventnum = col("Event#")
    c_stop = col("Stop")
    c_peak = col("Peak")

    c_goes = None
    for k, orig in cols_lower.items():
        if "goes" in k and "class" in k:
            c_goes = orig
            break

    c_pos = None
    for k, orig in cols_lower.items():
        if k.startswith("derived position"):
            c_pos = orig
            break

    rows: List[Dict[str, Any]] = []

    # Keep only flare rows and normalize them into one schema.
    for _, r in event_df.iterrows():
        ename = str(r.get(c_ename, "")).strip() if c_ename else ""
        start_str = str(r.get(c_start, "")).strip() if c_start else ""

        if not ename.startswith("gev_"):
            continue
        if start_str in {"", "-", "nan"}:
            continue

        start_dt = datetime.strptime(start_str, "%Y/%m/%d %H:%M:%S")
        date_part = start_str.split()[0]

        stop_str = str(r.get(c_stop, "")).strip() if c_stop else ""
        peak_str = str(r.get(c_peak, "")).strip() if c_peak else ""

        peak_dt = parse_datetime_or_none(date_part, peak_str)
        stop_dt = parse_datetime_or_none(date_part, stop_str)

        if stop_dt is not None and stop_dt < start_dt:
            stop_dt += timedelta(days=1)

        goes_class = str(r.get(c_goes, "")).strip() if c_goes else ""
        pos_text = str(r.get(c_pos, "")).strip() if c_pos else ""

        lat, lon, ar = parse_position(pos_text)

        event_num = None
        if c_eventnum:
            try:
                event_num = int(r.get(c_eventnum))
            except Exception:
                event_num = None

        rows.append(
            {
                "source_url": url,
                "event_number": event_num,
                "ename": ename,
                "event_start_utc": start_dt.isoformat(),
                "event_end_utc": stop_dt.isoformat() if stop_dt else "",
                "peak_utc": peak_dt.isoformat() if peak_dt else "",
                "flare_class": goes_class,
                "latitude": lat or "",
                "longitude": lon or "",
                "region_id": ar or "",
                "raw_position": pos_text,
            }
        )

    return rows, False


def extract_daily_links_from_archive():
    html, status = safe_get(ARCHIVE_URL)
    if html is None:
        raise RuntimeError("Could not download archive page.")

    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "last_events_" in href and href.endswith("/index.html"):
            links.append(urljoin(ARCHIVE_URL, href))

    return list(dict.fromkeys(links))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2020-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-12-31", help="YYYY-MM-DD")
    parser.add_argument("--out", type=str, default="data/lmsal_events_2020_2025.csv")
    args = parser.parse_args()

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()

    links = extract_daily_links_from_archive()
    print("Archive links found:", len(links))

    # Filter archive links to the requested date window.
    filtered: List[str] = []
    for u in links:
        d = get_date_from_last_events_url(u)
        if d and start_d <= d <= end_d:
            filtered.append(u)
    links = filtered

    print("Pages to scrape:", len(links))
    print("Starting scrape...")

    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    broken_path = os.path.join("output", "broken_links.txt")
    summary_path = os.path.join("output", "run_summary.txt")

    all_rows: List[Dict[str, Any]] = []
    broken_links: List[str] = []

    for i, url in enumerate(links, start=1):
        rows, broken = scrape_day_url(url)
        if broken:
            broken_links.append(url)

        all_rows.extend(rows)

        if i % 50 == 0 or i == len(links):
            print(
                f"[{i}/{len(links)}] pages processed | rows so far: {len(all_rows)} | broken: {len(broken_links)}"
            )

    df = pd.DataFrame(all_rows)

    before = len(df)
    # Drop duplicate events when the archive repeats entries.
    if before > 0:
        df = df.drop_duplicates(
            subset=["ename", "event_start_utc", "flare_class"],
            keep="first"
        ).reset_index(drop=True)

    df["event_start_utc"] = pd.to_datetime(df["event_start_utc"], errors="coerce", utc=True)
    df["event_end_utc"] = pd.to_datetime(df["event_end_utc"], errors="coerce", utc=True)
    df["peak_utc"] = pd.to_datetime(df["peak_utc"], errors="coerce", utc=True)
    df["region_id"] = df["region_id"].astype(str).str.strip()

    # Clip to the same calendar range as --start / --end (inclusive).
    ev_date = df["event_start_utc"].dt.tz_convert("UTC").dt.date
    df = df[(ev_date >= start_d) & (ev_date <= end_d)].copy()

    keep_cols = [
        "event_start_utc",
        "event_end_utc",
        "peak_utc",
        "flare_class",
        "region_id",
        "ename",
        "event_number",
        "source_url",
        "latitude",
        "longitude",
        "raw_position",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_csv(args.out, index=False)

    with open(broken_path, "w", encoding="utf-8") as f:
        for u in broken_links:
            f.write(u + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"pages_processed={len(links)}\n")
        f.write(f"rows_before_dedup={before}\n")
        f.write(f"rows_written={len(df)}\n")
        f.write(f"unique_enames={df['ename'].nunique() if len(df) else 0}\n")
        f.write(f"dupes_removed={before - len(df)}\n")
        f.write(f"broken_links={len(broken_links)}\n")

    print(f"Saved {len(df)} unique rows to {args.out}")
    print("Broken links logged to", broken_path)


if __name__ == "__main__":
    main()