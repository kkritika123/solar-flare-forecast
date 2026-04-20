"""Per-file and per-model orchestration: download -> parse -> yearly CSVs."""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set

from flare_scoreboard.crawl import crawl_model_files_smart, top_level_year_folders_on_server
from flare_scoreboard.csv_output import write_model_year_csvs_split
from flare_scoreboard.http_client import download_file, get_ext
from flare_scoreboard.parsers import (
    parse_iswa_xml,
    parse_json_forecast,
    parse_txt_or_xml_disguised_as_txt,
)


def process_one(url: str, model_name: str, raw_dir: str) -> List[Dict]:
    # Download once to local cache, then parse by extension.
    local = download_file(url, root=raw_dir)
    if not local:
        return []

    ext = os.path.splitext(local)[1].lower()

    if ext == ".xml":
        return parse_iswa_xml(local, model_name=model_name, source_url=url)
    if ext == ".txt":
        return parse_txt_or_xml_disguised_as_txt(
            local, model_name=model_name, source_url=url
        )
    if ext == ".json":
        return parse_json_forecast(local, model_name=model_name, source_url=url)

    return []


def process_model(
    model_url: str,
    years_set: Set[int],
    parse_exts: Set[str],
    raw_dir: str,
    out_dir: str,
    workers: int,
    download_all: bool,
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

    write_model_year_csvs_split(model_rows, model_name, out_dir, years_set=years_set)
    print(f"  Wrote yearly CSVs to: {os.path.join(out_dir, model_name)}")
