"""Parse XML / TXT / JSON forecast files into normalized row dicts."""
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from flare_scoreboard.parse_core import (
    issue_time_from_txt_basename_and_url,
    normalize_xml_flare_threshold,
    safe_probability,
)


def _preferred_region_id_from_subgroup(sg: ET.Element) -> str:
    """Prefer NOAA AR numbers (LMSAL events use region_id like 4325)."""
    noaa = ""
    fallback = ""
    for sr in sg.findall("./sourceregion"):
        id_el = sr.find("id")
        if id_el is None:
            continue
        t = (id_el.text or "").strip()
        if not t:
            continue
        scheme = (id_el.get("scheme") or "").lower()
        if "noaa_ar" in scheme:
            return t
        if not fallback:
            fallback = t
    alt = (sg.findtext("./region/id") or "").strip()
    if alt and not fallback:
        fallback = alt
    return noaa or fallback


def _prediction_window_bounds(fc: ET.Element) -> tuple:
    """ISWA files use starttime/endtime; some schemas use begin/end."""
    pw = fc.find("predictionwindow")
    if pw is None:
        b = (
            fc.findtext("predictionwindow/begin")
            or fc.findtext("predictionwindow/starttime")
            or ""
        ).strip()
        e = (
            fc.findtext("predictionwindow/end")
            or fc.findtext("predictionwindow/endtime")
            or ""
        ).strip()
        return b, e
    begin = (
        (pw.findtext("begin") or "").strip()
        or (pw.findtext("starttime") or "").strip()
        or (pw.findtext("start") or "").strip()
    )
    end = (pw.findtext("end") or "").strip() or (pw.findtext("endtime") or "").strip()
    return begin, end


def _append_entry_row(
    rows: List[Dict],
    entry: ET.Element,
    model_name: str,
    method: str,
    issue_time: str,
    begin: str,
    end: str,
    ftype: str,
    region_id: str,
    source_url: str,
    xml_path: str,
) -> None:
    fluxbin = entry.find("fluxbin")
    threshold = normalize_xml_flare_threshold(
        (fluxbin.get("name") if fluxbin is not None else "") or ""
    )

    val_text = (
        entry.findtext("probability/value") or entry.findtext("value") or ""
    ).strip()
    if not val_text:
        return

    prob = safe_probability(val_text)
    if prob is None:
        return

    rows.append(
        {
            "model_name": model_name,
            "method": method,
            "forecast_type": ftype,
            "issue_time_utc": issue_time,
            "runtime_utc": issue_time,
            "window_begin_utc": begin,
            "window_end_utc": end,
            "flare_threshold": threshold,
            "probability": prob,
            "region_id": region_id if ftype == "region" else "",
            "source_url": source_url,
            "source_file": xml_path,
        }
    )


def parse_iswa_xml(xml_path: str, model_name: str, source_url: str) -> List[Dict]:
    rows = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return rows

    forecasts = root.findall(".//forecast")
    if not forecasts:
        return rows

    for fc in forecasts:
        method = (fc.findtext("method") or model_name).strip()
        issue_time = (fc.findtext("issuetime") or "").strip()
        begin, end = _prediction_window_bounds(fc)

        emitted = False
        for g in fc.findall("./group"):
            ft_el = g.find("forecasttype")
            if ft_el is None:
                continue
            ftype_raw = (ft_el.text or "").strip().lower()
            ftype = "region" if ftype_raw == "region" else "full_disk"

            if ftype == "full_disk":
                entries = g.findall("./entry")
                if entries:
                    emitted = True
                for entry in entries:
                    _append_entry_row(
                        rows,
                        entry,
                        model_name,
                        method,
                        issue_time,
                        begin,
                        end,
                        ftype,
                        "",
                        source_url,
                        xml_path,
                    )
            else:
                for sg in g.findall("./group"):
                    region_id = _preferred_region_id_from_subgroup(sg)
                    sg_entries = sg.findall("./entry")
                    if sg_entries:
                        emitted = True
                    for entry in sg_entries:
                        _append_entry_row(
                            rows,
                            entry,
                            model_name,
                            method,
                            issue_time,
                            begin,
                            end,
                            ftype,
                            region_id,
                            source_url,
                            xml_path,
                        )

        if not emitted:
            forecast_type_text = (
                fc.findtext("group/forecasttype")
                or fc.findtext(".//forecasttype")
                or ""
            ).strip()
            ftype = "region" if forecast_type_text.lower() == "region" else "full_disk"

            region_id = ""
            if ftype == "region":
                for p in [
                    "group/sourceregion/id",
                    "sourceregion/id",
                    ".//sourceregion/id",
                    ".//region/id",
                ]:
                    t = (fc.findtext(p) or "").strip()
                    if t:
                        region_id = t
                        break

            for entry in fc.findall(".//entry"):
                _append_entry_row(
                    rows,
                    entry,
                    model_name,
                    method,
                    issue_time,
                    begin,
                    end,
                    ftype,
                    region_id,
                    source_url,
                    xml_path,
                )

    return rows


def _assa_normalize_iso(ts: str) -> str:
    """ASSA uses e.g. 2024-01-01T04:00Z — align with other parsers (no Z, with seconds)."""
    s = (ts or "").strip().replace("Z", "").replace("z", "")
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", s):
        return f"{s}:00"
    return s


def _assa_meta(
    txt: str, basename: str, source_url: str, model_name: str
) -> Dict[str, str]:
    """Issue time and prediction window from ASSA header; fallback to filename/URL."""
    issue = ""
    m = re.search(
        r"(?m)^\s*#?\s*Issue Time:\s*(\S+)", txt, re.IGNORECASE
    )
    if m:
        issue = _assa_normalize_iso(m.group(1))
    if not issue:
        issue = issue_time_from_txt_basename_and_url(basename, source_url)

    w0 = w1 = ""
    m = re.search(
        r"(?m)^\s*#?\s*Prediction Window Start Time:\s*(\S+)", txt, re.IGNORECASE
    )
    if m:
        w0 = _assa_normalize_iso(m.group(1))
    m = re.search(
        r"(?m)^\s*#?\s*Prediction Window End Time:\s*(\S+)", txt, re.IGNORECASE
    )
    if m:
        w1 = _assa_normalize_iso(m.group(1))

    method = model_name
    m = re.search(
        r"(?m)^\s*#?\s*Forecasting Method:\s*(.+)$", txt, re.IGNORECASE
    )
    if m:
        method = m.group(1).strip()

    return {"issue_time": issue, "window_begin": w0, "window_end": w1, "method": method}


def parse_assa_txt(txt_path: str, model_name: str, source_url: str) -> List[Dict]:
    """
    Tabular scoreboard .txt: **ASSA**, **AMOS_v1**, **SPS** (``#`` on headers optional),
    etc. ``Full Disk Forecast`` / ``#Full Disk Forecast`` plus optional region block.
    NOAA AR may be ``----`` (ASSA) or numeric (AMOS); ``region_id`` set only for
    numeric NOAA ids.

    Without this parser, many of these files are mis-read by ``parse_txt_generic``.
    """
    rows: List[Dict] = []
    base = os.path.basename(txt_path)
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return rows

    meta = _assa_meta(txt, base, source_url, model_name)
    issue = meta["issue_time"]
    wb, we = meta["window_begin"], meta["window_end"]
    method = meta["method"]

    def append_row(ftype: str, thr: str, prob: float, region_id: str) -> None:
        p = safe_probability(prob)
        if p is None:
            return
        t = normalize_xml_flare_threshold(thr)
        if t not in ("C", "M", "X"):
            return
        rows.append(
            {
                "model_name": model_name,
                "method": method,
                "forecast_type": ftype,
                "issue_time_utc": issue,
                "runtime_utc": issue,
                "window_begin_utc": wb,
                "window_end_utc": we,
                "flare_threshold": t,
                "probability": p,
                "region_id": region_id if ftype == "region" else "",
                "source_url": source_url,
                "source_file": txt_path,
            }
        )

    # --- Full disk: data line has 3+ floats; skip title/header (X_prob line, etc.)
    fd_m = re.search(r"(?m)^\s*#?\s*Full Disk Forecast\b", txt, re.IGNORECASE)
    if fd_m:
        chunk = txt[fd_m.start() : fd_m.start() + 4000]
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        data_line = None
        for ln in lines[1:]:
            if "prob" in ln.lower():
                continue
            floats = [float(x) for x in re.findall(r"\d+\.\d+", ln)]
            if len(floats) >= 3:
                data_line = ln
                break
        if data_line is not None:
            floats = [float(x) for x in re.findall(r"\d+\.\d+", data_line)]
            x_p, m_p, c_p = floats[0], floats[1], floats[2]
            append_row("full_disk", "X", x_p, "")
            append_row("full_disk", "M", m_p, "")
            append_row("full_disk", "C", c_p, "")

    # --- Region: lines like " 1 2024-01-01T00:00Z N76E08 ---- ..."
    rg_m = re.search(r"(?m)^\s*#?\s*Region Forecast\b", txt, re.IGNORECASE)
    if rg_m:
        rchunk = txt[rg_m.start() :]
        for ln in rchunk.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            m = re.match(
                r"^(\d+)\s+(\d{4}-\d{2}-\d{2}T[^\s]+)\s+(\S+)\s+(\S+)",
                s,
            )
            if not m:
                continue
            ar_tok = m.group(4)
            region_id = ar_tok if re.fullmatch(r"\d{3,5}", ar_tok) else ""

            floats = [float(x) for x in re.findall(r"\d+\.\d+", s)]
            if len(floats) < 3:
                continue
            x_p, m_p, c_p = floats[0], floats[1], floats[2]
            append_row("region", "X", x_p, region_id)
            append_row("region", "M", m_p, region_id)
            append_row("region", "C", c_p, region_id)

    return rows


def _looks_like_assa_txt(chunk: str) -> bool:
    if re.search(r"Forecasting Method:\s*ASSA\b", chunk, re.IGNORECASE):
        return True
    if not re.search(r"(?m)^\s*#?\s*Full Disk Forecast\b", chunk, re.IGNORECASE):
        return False
    if re.search(r"(?m)^\s*#?\s*Region Forecast\b", chunk, re.IGNORECASE):
        return True
    # SPS / ASSA-style metadata (# optional on each line)
    if re.search(r"(?m)^\s*#?\s*Issue Time:", chunk, re.IGNORECASE):
        return True
    if re.search(r"(?m)^\s*#?\s*Forecasting\s+method:", chunk, re.IGNORECASE):
        return True
    # Table present without matching Issue Time line in first 16k (unlikely but safe)
    if re.search(r"\bX_prob\b", chunk, re.IGNORECASE):
        return True
    return False


def parse_txt_or_xml_disguised_as_txt(
    txt_path: str, model_name: str, source_url: str
) -> List[Dict]:
    """
    NASA ISWA often serves XML payloads with a .txt extension (e.g. SIDC_v2).
    Those must use the XML parser so prediction windows and region AR ids are kept.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(16384)
    except OSError:
        return []

    head = chunk.lstrip()
    if head.startswith("<?xml") or re.search(
        r"<forecast\b", chunk[:12000], re.IGNORECASE
    ):
        return parse_iswa_xml(txt_path, model_name=model_name, source_url=source_url)

    if _looks_like_assa_txt(chunk):
        return parse_assa_txt(txt_path, model_name=model_name, source_url=source_url)

    return parse_txt_generic(txt_path, model_name=model_name, source_url=source_url)


def parse_txt_generic(txt_path: str, model_name: str, source_url: str) -> List[Dict]:
    rows = []
    base = os.path.basename(txt_path)

    issue_time = issue_time_from_txt_basename_and_url(base, source_url)

    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return rows

    ftype = "full_disk"
    region_id = ""

    if re.search(r"\bregion\b", txt, re.IGNORECASE) or re.search(
        r"\bAR\s*[:#]?\s*(\d{4,6})\b", txt, re.IGNORECASE
    ):
        ftype = "region"
        m_ar = re.search(r"\bAR\s*[:#]?\s*(\d{4,6})\b", txt, re.IGNORECASE)
        if m_ar:
            region_id = m_ar.group(1)

    bins = ["C", "M", "X", "M1+", "M5+", "X1+", "X5+"]

    for b in bins:
        prob = None

        m_pct = re.search(
            rf"\b{re.escape(b)}\b[^0-9%]*([0-9]+(?:\.[0-9]+)?)\s*%",
            txt,
            re.IGNORECASE,
        )
        if m_pct:
            prob = float(m_pct.group(1)) / 100.0
        else:
            m_dec = re.search(
                rf"\b{re.escape(b)}\b[^0-9.]*([01](?:\.\d+)?)",
                txt,
                re.IGNORECASE,
            )
            if m_dec:
                prob = float(m_dec.group(1))

        if prob is None:
            continue

        rows.append(
            {
                "model_name": model_name,
                "method": model_name,
                "forecast_type": ftype,
                "issue_time_utc": issue_time,
                "runtime_utc": issue_time,
                "window_begin_utc": "",
                "window_end_utc": "",
                "flare_threshold": b,
                "probability": prob,
                "region_id": region_id if ftype == "region" else "",
                "source_url": source_url,
                "source_file": txt_path,
            }
        )

    return rows


def parse_json_forecast(json_path: str, model_name: str, source_url: str) -> List[Dict]:
    rows = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return rows

    fs = data.get("forecast_submission", {})
    issue_time = fs.get("issue_time", "")

    def _json_region_id(fc: dict) -> str:
        """Best-effort region id from JSON forecast payloads."""
        rid = str(fc.get("source_region", "") or fc.get("region_id", "") or "").strip()
        if rid:
            return rid

        # DAFFS-style payload: region_ids: [{type: NOAA|HARP, number: "..."}]
        ids = fc.get("region_ids")
        if isinstance(ids, list):
            # Prefer NOAA number when available.
            for item in ids:
                if not isinstance(item, dict):
                    continue
                n = str(item.get("number", "")).strip()
                t = str(item.get("type", "")).strip().upper()
                if n and t == "NOAA":
                    return n
            # Fallback: first numeric-looking "number" in the list.
            for item in ids:
                if not isinstance(item, dict):
                    continue
                n = str(item.get("number", "")).strip()
                if n:
                    return n
        return ""

    def parse_block(forecast_list, ftype: str):
        out = []
        for fc in forecast_list or []:
            win = fc.get("prediction_window", {})
            begin = win.get("start_time", "")
            end = win.get("end_time", "")

            region_id = ""
            if ftype == "region":
                region_id = _json_region_id(fc)

            for p in fc.get("flare_probabilities", []) or []:
                thr = str(p.get("class", "")).upper()
                prob = safe_probability(p.get("probability"))

                if prob is None:
                    continue

                out.append(
                    {
                        "model_name": model_name,
                        "method": model_name,
                        "forecast_type": ftype,
                        "issue_time_utc": issue_time,
                        "runtime_utc": issue_time,
                        "window_begin_utc": begin,
                        "window_end_utc": end,
                        "flare_threshold": thr,
                        "probability": prob,
                        "region_id": region_id if ftype == "region" else "",
                        "source_url": source_url,
                        "source_file": json_path,
                    }
                )
        return out

    rows.extend(parse_block(fs.get("full_disk_forecasts", []), "full_disk"))
    rows.extend(parse_block(fs.get("region_forecasts", []), "region"))
    return rows
