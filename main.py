
"""Download CCMC flare forecasts and write CSVs per model."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from flare_scoreboard import (
    load_config,
    normalize_dir,
    discover_models,
    process_model,
)


def _model_name(model_url: str) -> str:
    return model_url.rstrip("/").split("/")[-1]


def _parse_date(value: str):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {value}. Use YYYY-MM-DD."
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(
        description="Download and process solar flare forecasts."
    )
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    args = parser.parse_args()

    if (args.start is None) != (args.end is None):
        parser.error("Provide both --start and --end.")

    if args.start and args.start >= args.end:
        parser.error("--start must be earlier than --end.")

    cfg = load_config()

    base_url = normalize_dir(cfg["base_url"])
    years_set = cfg["years_set"]
    parse_exts = cfg["parse_exts_set"]
    raw_dir = cfg["raw_dir"]
    out_dir = cfg["out_dir"]
    workers = cfg["workers"]
    download_all = cfg["download_all_files"]
    models_filter = cfg.get("models_filter")

    if args.start is not None:
        # Discover all year folders touched by the requested interval.
        # The end boundary is exclusive.
        from datetime import timedelta

        last_included = args.end - timedelta(microseconds=1)
        years_set = set(
            range(args.start.year, last_included.year + 1)
        )

        date_label = (
            f"{args.start.strftime('%Y%m%dT%H%M%S')}_"
            f"{args.end.strftime('%Y%m%dT%H%M%S')}"
        )

        out_dir = str(Path(out_dir) / "date_ranges" / date_label)

        print(f"Date range: {args.start} to {args.end}")
        print(f"Filtered output: {out_dir}")

    print("Discovering model folders...")
    models = discover_models(base_url)

    print(f"Models found: {len(models)}")
    for model in models:
        print(f" - {_model_name(model)}")

    if models_filter is not None:
        label = cfg.get("models_filter_label") or "models"

        if not models_filter:
            print(f"\nConfig model list {label!r} is empty.")
            return

        discovered = {_model_name(u) for u in models}
        before = len(models)

        models = [
            u for u in models
            if _model_name(u) in models_filter
        ]

        print(
            f"\nFiltered by {label!r}: "
            f"{len(models)} of {before} folder(s) "
            "will be processed."
        )

        not_found = sorted(models_filter - discovered)
        if not_found:
            print(
                "  [WARN] Not found on server:",
                ", ".join(not_found),
            )

    for model_url in models:
        process_model(
            model_url=model_url,
            years_set=years_set,
            parse_exts=parse_exts,
            raw_dir=raw_dir,
            out_dir=out_dir,
            workers=workers,
            download_all=download_all,
            start=args.start,
            end=args.end,
        )

    print(f"\nDone. Outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
