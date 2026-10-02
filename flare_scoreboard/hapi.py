"""Retrieve forecast records from the CCMC Flare Scoreboard HAPI API."""

from io import StringIO

from attrs import fields
from matplotlib.style import available
import pandas as pd
import requests


HAPI_URL = (
    "https://iswa.ccmc.gsfc.nasa.gov/"
    "IswaSystemWebApp/flarescoreboard/hapi"
)


def _get_json(endpoint, params):
    response = requests.get(
        f"{HAPI_URL}/{endpoint}",
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    status = payload.get("status", {})
    code = status.get("code", 1200)

    if code != 1200:
        raise ValueError(
            f"HAPI error {code}: {status.get('message', 'Unknown error')}"
        )

    return payload


def _utc_time(value):
    timestamp = pd.Timestamp(value)

    if pd.isna(timestamp):
        raise ValueError("A valid start/end date is required.")

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def _download_hapi_chunk(
    model_id,
    model_type,
    flare_type,
    start,
    end,
):
    """Return HAPI records for the requested issue-time range.

    start is inclusive; end is exclusive.
    Bare dates are interpreted as midnight UTC.

    flare_type selects an exact HAPI field, such as M or MPlus.
    These fields are not automatically treated as equivalent.
    """
    type_mapping = {
        "full_disk": "FULLDISK",
        "fulldisk": "FULLDISK",
        "active_region": "REGIONS",
        "regions": "REGIONS",
    }

    model_type = model_type.lower()

    if model_type not in type_mapping:
        raise ValueError(
            "model_type must be 'full_disk' or 'active_region'."
        )

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string.")

    # A-Effort uses a different spelling in HAPI dataset IDs.
    hapi_model = {
        "A-Effort": "AEffort",
    }.get(model_id, model_id)

    dataset_id = f"{hapi_model}_{type_mapping[model_type]}"

    start_time = _utc_time(start)
    end_time = _utc_time(end)

    if start_time >= end_time:
        raise ValueError("start must be earlier than end.")

    # Check which fields this dataset supports.
    info = _get_json(
        "info",
        {"id": dataset_id, "options": "fields.supported"},
    )

    metadata = info.get("parameters", [])
    available = {field["name"]: field for field in metadata}

    if flare_type not in available:
        raise ValueError(
            f"{dataset_id} does not support {flare_type!r}. "
            f"Available fields: {', '.join(available)}"
        )

    fields = ["start_window", "end_window", "issue_time", flare_type]

    if type_mapping[model_type] == "REGIONS":
        fields.append("NOAARegionId")

    missing = [name for name in fields if name not in available]
    if missing:
        raise ValueError(
            f"{dataset_id} is missing required fields: {missing}"
        )

    # Use metadata for the same field selection as the DATA request.
    selected_info = _get_json(
        "info",
        {
            "id": dataset_id,
            "parameters": ",".join(fields),
            "options": "fields.supported",
        },
    )

    # Match the exact fields requested from /data.
    selected_fields = [available[name] for name in fields]
    column_names = fields.copy()
    
    response = requests.get(
        f"{HAPI_URL}/data",
        params={
            "id": dataset_id,
            "time.min": start_time.isoformat().replace("+00:00", "Z"),
            "time.max": end_time.isoformat().replace("+00:00", "Z"),
            "parameters": ",".join(fields),
            "format": "csv",
            "options": "fields.supported",
        },
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(
            f"HAPI DATA request failed: HTTP {response.status_code}\n"
            f"Server response: {response.text[:2000]}"
    )

    # HAPI may return a JSON status instead of CSV.
    if response.text.lstrip().startswith("{"):
        payload = response.json()
        status = payload.get("status", {})
        if status.get("code") == 1201:
            return pd.DataFrame(columns=column_names)
        raise ValueError(f"HAPI DATA response: {status}")

    if not response.text.strip():
        return pd.DataFrame(columns=column_names)

    df = pd.read_csv(
        StringIO(response.text),
        header=None,
        comment="#",
    )

    if len(df.columns) != len(column_names):
        print("INFO column names:", column_names)
        print("CSV column count:", len(df.columns))
        print("First CSV rows:")
        print(response.text[:1000])
        raise ValueError("HAPI CSV columns do not match INFO metadata.")

    df.columns = column_names

    # Replace documented missing-value markers.
    for field in selected_fields:
        fill = field.get("fill")
        if fill is not None:
            df[field["name"]] = df[field["name"]].replace(
                [fill, str(fill)], pd.NA
            )

    for name in ("start_window", "end_window", "issue_time"):
        df[name] = pd.to_datetime(df[name], utc=True, errors="coerce")

    df[flare_type] = pd.to_numeric(df[flare_type], errors="coerce")

    df.attrs["dataset_id"] = dataset_id
    df.attrs["request_url"] = response.url

    return df

def download_hapi(model_id, model_type, flare_type, start, end):
    """Download forecasts selected by window start.

    start is inclusive; end is exclusive. Dates use UTC.
    Requests are split into ranges of at most 31 days.
    """
    start_time = _utc_time(start)
    end_time = _utc_time(end)

    if start_time >= end_time:
        raise ValueError("start must be earlier than end.")

    frames = []
    query_urls = []
    current = start_time

    while current < end_time:
        chunk_end = min(
            current + pd.Timedelta(days=31),
            end_time,
        )

        print(f"Downloading {current.date()} to {chunk_end.date()}")

        frame = _download_hapi_chunk(
            model_id=model_id,
            model_type=model_type,
            flare_type=flare_type,
            start=current,
            end=chunk_end,
        )

        query_urls.append(frame.attrs.get("request_url"))

        # Keep metadata separately because each request has a different URL.
        frame.attrs = {}
        frames.append(frame)

        current = chunk_end

    result = pd.concat(frames, ignore_index=True)

    # Keep forecast windows within [start, end).
    result = result.loc[
        (result["start_window"] >= start_time)
        & (result["start_window"] < end_time)
    ].copy()

    # Remove identical rows repeated at request boundaries.
    result = result.drop_duplicates().reset_index(drop=True)
    result.attrs["request_urls"] = [
        url for url in query_urls if url is not None
    ]

    return result