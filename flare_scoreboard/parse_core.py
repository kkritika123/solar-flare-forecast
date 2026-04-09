# ── Small helpers (time strings, probabilities, filenames) ────────────────────
import os
import re
from datetime import datetime
from typing import Optional


def year_from_issue_time(issue_time: str) -> Optional[int]:
    if not issue_time:
        return None
    s = issue_time.replace("Z", "")
    try:
        return datetime.fromisoformat(s).year
    except ValueError:
        return None


def normalize_xml_flare_threshold(fluxbin_name: str) -> str:
    """ISWA fluxbin names are often C+/M+/X+; evaluation uses single-letter C/M/X."""
    s = (fluxbin_name or "").strip().upper()
    if not s:
        return "unknown"
    if s[0] in ("C", "M", "X"):
        return s[0]
    return s


def safe_probability(value):
    try:
        p = float(value)
        if p > 1:
            p = p / 100.0
        return p
    except (TypeError, ValueError):
        return None


def issue_time_from_txt_basename_and_url(basename: str, source_url: str) -> str:
    """
    Many scoreboard .txt files are not named YYYYMMDD_HHMMSS. Derive issue time from
    the filename and, if needed, from .../YYYY/MM/... in the URL so rows get a year when
    writing yearly CSVs.
    """
    base = os.path.basename(basename)

    # Try most specific filename patterns first, then fall back to URL folder dates.
    m = re.search(r"(\d{8})[_-](\d{6})", base)
    if m:
        return (
            f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
            f"T{m.group(2)[:2]}:{m.group(2)[2:4]}:{m.group(2)[4:6]}"
        )

    m = re.search(r"(\d{8})[_-](\d{4})", base)
    if m:
        return (
            f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
            f"T{m.group(2)[:2]}:{m.group(2)[2:4]}:00"
        )

    m = re.search(r"(?<![0-9])(\d{4})-(\d{2})-(\d{2})(?![0-9])", base)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00"

    m = re.search(
        r"(?<![0-9])((?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?![0-9])",
        base,
    )
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}T12:00:00"

    url = source_url.replace("\\", "/")
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/[^/]+$", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00"

    m = re.search(r"/(\d{4})/(\d{2})/[^/]+\.txt$", url, re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01T00:00:00"

    return ""
