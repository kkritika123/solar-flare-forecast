"""Per-file and per-model orchestration: download -> parse -> yearly CSVs."""
from datetime import datetime, timezone
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set
from flare_scoreboard.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from flare_scoreboard.crawl import crawl_model_files_smart, top_level_year_folders_on_server
from flare_scoreboard.csv_output import write_model_year_csvs_split
from flare_scoreboard.http_client import download_file, get_ext
from flare_scoreboard.parsers import (
    parse_iswa_xml,
    parse_json_forecast,
    parse_txt_or_xml_disguised_as_txt,
)



def process_one(url: str, model_name: str, raw_dir: str) -> List[Dict]:
    """Download, parse, and checkpoint one forecast file."""

    local = download_file(url, root=raw_dir)

    if not local:
        return []

    cache_dir = os.path.join(raw_dir, ".checkpoints")

    cached_rows = load_checkpoint(
        url,
        cache_dir,
        local,
        model_name=model_name,
    )

    if cached_rows is not None:
        return cached_rows

    ext = os.path.splitext(local)[1].lower()

    if ext == ".xml":
        rows = parse_iswa_xml(
            local,
            model_name=model_name,
            source_url=url,
        )
    elif ext == ".txt":
        rows = parse_txt_or_xml_disguised_as_txt(
            local,
            model_name=model_name,
            source_url=url,
        )
    elif ext == ".json":
        rows = parse_json_forecast(
            local,
            model_name=model_name,
            source_url=url,
        )
    else:
        return []

    if rows:
        save_checkpoint(
            url,
            cache_dir,
            local,
            rows,
            model_name=model_name,
        )

    return rows
def filter_rows_by_date(rows, start=None, end=None):
    """Keep forecasts issued within the requested UTC date range.

    The start is inclusive and the end is exclusive.
    """
    if start is None and end is None:
        return rows

    def parse_date(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(
                str(value).strip().replace("Z", "+00:00")
            )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    start_dt = parse_date(start)
    end_dt = parse_date(end)

    if start_dt and end_dt and start_dt >= end_dt:
        raise ValueError("start must be earlier than end")

    filtered = []

    for row in rows:
        issue_time = row.get("issue_time_utc")
        if not issue_time:
            continue

        try:
            issued = parse_date(issue_time)
        except (ValueError, TypeError):
            continue

        if start_dt is not None and issued < start_dt:
            continue
        if end_dt is not None and issued >= end_dt:
            continue

        filtered.append(row)

    return filtered

def process_model(
    model_url: str,
    years_set: Set[int],
    parse_exts: Set[str],
    raw_dir: str,
    out_dir: str,
    workers: int,
    download_all: bool,
    start=None,
    end=None,
):
    model_name = model_url.rstrip("/").split("/")[-1]
    print(f"\n=== Model: {model_name} ===")

    try:
        file_urls = crawl_model_files_smart(model_url, years_set)
    except Exception as e:
        print(f"  [MODEL SKIPPED] {model_name}  error={e}")
        return

    if not file_urls:
        ys = ", ".join(str(y) for y in sorted(years_set))
        print(
            f"  [INFO] No files under year folders {{{ys}}} for this model on the server. "
            "Some scoreboard entries only publish older years (e.g. 2015–2016) — check the model’s directory listing online."
        )
        on_server = top_level_year_folders_on_server(model_url)
        if on_server:
            overlap = sorted(years_set.intersection(on_server))
            print(
                f"  [INFO] Year folders that exist at this model’s root: {on_server}. "
                f"Overlap with your config: {overlap if overlap else 'none — add those years to config \"years\" if you need them'}."
            )

    ext_count = {}
    for u in file_urls:
        e = get_ext(u)
        ext_count[e] = ext_count.get(e, 0) + 1

    print("  Top extensions:")
    for e in sorted(ext_count, key=ext_count.get, reverse=True)[:10]:
        print(f"   - {e}: {ext_count[e]}")

    # Keep only configured parse extensions unless download_all_files is enabled.
    if download_all:
        work_urls = file_urls
    else:
        work_urls = [u for u in file_urls if get_ext(u) in parse_exts]

    work_urls.sort()
    print("  Files to process:", len(work_urls))

    model_rows = []
    produced_files = 0

    # Parse files in parallel; each worker returns rows for one source file.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(
            ex.map(lambda u: process_one(u, model_name, raw_dir), work_urls), 1
        ):
            if rows:
                produced_files += 1
                model_rows.extend(rows)

            if i % 500 == 0 or i == len(work_urls):
                print(f"  Processed {i}/{len(work_urls)} files...")

    print("  Files that produced parsed rows:", produced_files)
    print("  Total parsed rows:", len(model_rows))

    model_rows = filter_rows_by_date(model_rows, start=start, end=end)

    print("  Rows after date filtering:", len(model_rows))

    write_model_year_csvs_split(model_rows, model_name, out_dir, years_set=years_set)
