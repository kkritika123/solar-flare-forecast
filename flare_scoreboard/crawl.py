"""Discover model folders and crawl the CCMC directory tree."""
from typing import List, Set

from flare_scoreboard.http_client import list_dir, normalize_dir


def discover_models(base_url: str) -> List[str]:
    base_url = normalize_dir(base_url)
    dirs, _ = list_dir(base_url)

    models = []
    for d in dirs:
        d = normalize_dir(d)
        if d.startswith(base_url):
            models.append(d)

    models.sort()
    return models


def top_level_year_folders_on_server(model_url: str) -> List[int]:
    """4-digit year directory names directly under the model root (e.g. 2024/, 2016/)."""
    try:
        dirs, _ = list_dir(normalize_dir(model_url))
    except Exception:
        return []
    found: List[int] = []
    for d in dirs:
        last = normalize_dir(d).rstrip("/").split("/")[-1]
        if last.isdigit() and len(last) == 4:
            found.append(int(last))
    return sorted(set(found))


def crawl_model_files_smart(
    model_url: str, years_set: Set[int], max_find_depth: int = 4
) -> List[str]:
    model_url = normalize_dir(model_url)
    visited = set()
    year_dirs = []

    def find_year_dirs(url: str, depth: int):
        # First pass: locate year folders (e.g. /2024/) under the model tree.
        url = normalize_dir(url)

        if url in visited:
            return
        visited.add(url)

        if depth > max_find_depth:
            return

        dirs, _ = list_dir(url)

        for d in dirs:
            d = normalize_dir(d)
            last = d.rstrip("/").split("/")[-1]

            if last.isdigit() and len(last) == 4:
                y = int(last)
                if y in years_set:
                    year_dirs.append(d)
                continue

            if d.startswith(model_url):
                find_year_dirs(d, depth + 1)

    find_year_dirs(model_url, 0)

    def collect_files_under(url: str, depth: int, max_depth: int = 3) -> List[str]:
        # Second pass: collect files under each selected year folder.
        url = normalize_dir(url)
        dirs, files = list_dir(url)
        out = list(files)

        if depth >= max_depth:
            return out

        for d in dirs:
            d = normalize_dir(d)
            out.extend(collect_files_under(d, depth + 1, max_depth))

        return out

    all_files = []
    for ydir in sorted(set(year_dirs)):
        all_files.extend(collect_files_under(ydir, 0, max_depth=3))

    return sorted(set(all_files))
