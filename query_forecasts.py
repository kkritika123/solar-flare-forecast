import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flare_scoreboard import get_predictions
from flare_scoreboard.hapi import _get_json, _utc_time


DEFAULT_MODELS = [
    "NOAA_1", "SIDC_v2", "ASSA_1", "ASSA_24H_1",
    "AMOS_v1", "ASAP_1", "A-Effort", "DAFFS", "MagPy", "SPS_1",
]

MODEL_TYPES = {
    "full_disk": "FULLDISK",
    "active_region": "REGIONS",
}


def main():
    parser = argparse.ArgumentParser(
        description="Query HAPI forecasts or processed archive CSVs."
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--type",
        choices=list(MODEL_TYPES),
        default=None,
    )
    parser.add_argument("--flare")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--source",
        choices=["hapi", "archive"],
        default="hapi",
    )
    parser.add_argument("--archive-dir", default="output")

    args = parser.parse_args()

    if args.models and not args.all:
        parser.error("--models requires --all.")

    if args.all and (args.model or args.out):
        parser.error(
            "--all cannot be combined with --model or --out."
        )

    if not args.all and (not args.model or not args.flare):
        parser.error(
            "Provide --model and --flare, or use --all."
        )

    try:
        start = _utc_time(args.start)
        end = _utc_time(args.end)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    if start >= end:
        parser.error("--start must be earlier than --end.")

    output_dir = Path("data") / args.source
    output_dir.mkdir(parents=True, exist_ok=True)

    date_label = (
        f"{start.strftime('%Y%m%dT%H%M%S')}_"
        f"{end.strftime('%Y%m%dT%H%M%S')}"
    )
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    report = []
    report_file = output_dir / (
        f"download_report_{date_label}_{run_id}.csv"
    )

    def record(
        model,
        model_type,
        flare,
        status,
        count=0,
        file="",
        error="",
    ):
        report.append({
            "model": model,
            "model_type": model_type,
            "flare_field": flare,
            "source": args.source,
            "status": status,
            "records": count,
            "file": str(file),
            "error": error,
        })
        pd.DataFrame(report).to_csv(
            report_file, index=False
        )

    def download(
        model,
        model_type,
        flare,
        output_name=None,
    ):
        filename = output_name or (
            f"{model}_{model_type}_{flare}_{date_label}.csv"
        )
        output_file = output_dir / Path(filename).name

        try:
            df = get_predictions(
                model_id=model,
                model_type=model_type,
                flare_type=flare,
                start=args.start,
                end=args.end,
                source=args.source,
                archive_dir=args.archive_dir,
            )

            df.to_csv(output_file, index=False)
            print(
                f"Saved {len(df)} records to {output_file}"
            )

            for url in df.attrs.get("request_urls", []):
                print(url)

            print(df.head().to_string(index=False))

            record(
                model,
                model_type,
                flare,
                "saved" if not df.empty else "empty",
                count=len(df),
                file=output_file,
            )

        except Exception as exc:
            print(
                f"Failed: {model}/{model_type}/{flare}: {exc}"
            )
            record(
                model,
                model_type,
                flare,
                "query_failed",
                error=str(exc),
            )

    if not args.all:
        download(
            args.model,
            args.type or "full_disk",
            args.flare,
            args.out,
        )

    else:
        selected_types = (
            {args.type: MODEL_TYPES[args.type]}
            if args.type
            else MODEL_TYPES
        )

        for model in (args.models or DEFAULT_MODELS):
            hapi_model = (
                "AEffort" if model == "A-Effort" else model
            )

            for model_type, suffix in selected_types.items():
                dataset_id = f"{hapi_model}_{suffix}"
                print(
                    f"\nChecking {model}/{model_type} "
                    f"from {args.source}"
                )

                try:
                    if args.source == "hapi":
                        info = _get_json(
                            "info",
                            {
                                "id": dataset_id,
                                "options": "fields.supported",
                            },
                        )

                        fields = [
                            field["name"]
                            for field in info.get("parameters", [])
                            if str(
                                field.get("units", "")
                            ).lower() == "probability"
                            and field["name"].upper().startswith(
                                ("C", "M", "X")
                            )
                        ]

                    else:
                        archive_type = (
                            "full_disk"
                            if model_type == "full_disk"
                            else "region"
                        )
                        folder = Path(args.archive_dir) / model
                        files = sorted(
                            folder.glob(
                                f"*_{archive_type}.csv"
                            )
                        )

                        if not files:
                            print(
                                f"No local {model_type} "
                                f"files for {model}"
                            )
                            record(
                                model,
                                model_type,
                                "",
                                "no_local_files",
                            )
                            continue

                        thresholds = set()

                        for file in files:
                            metadata = pd.read_csv(
                                file,
                                usecols=[
                                    "model_name",
                                    "forecast_type",
                                    "flare_threshold",
                                ],
                            )

                            matching = metadata.loc[
                                (metadata["model_name"] == model)
                                & (
                                    metadata["forecast_type"]
                                    == archive_type
                                )
                            ]

                            thresholds.update(
                                matching["flare_threshold"]
                                .dropna()
                                .astype(str)
                            )

                        fields = sorted(thresholds)

                except Exception as exc:
                    print(
                        f"Could not inspect "
                        f"{model}/{model_type}: {exc}"
                    )
                    record(
                        model,
                        model_type,
                        "",
                        "inspection_failed",
                        error=str(exc),
                    )
                    continue

                if not fields:
                    record(
                        model,
                        model_type,
                        "",
                        "no_probability_fields",
                    )
                    continue

                if args.flare:
                    if args.flare not in fields:
                        print(
                            f"Skipping {model}/{model_type}: "
                            f"field {args.flare} is unsupported."
                        )
                        record(
                            model,
                            model_type,
                            args.flare,
                            "unsupported_field",
                        )
                        continue

                    fields = [args.flare]

                for flare in fields:
                    download(model, model_type, flare)

    print(f"\nReport: {report_file}")

    if any(
        row["status"] in {
            "query_failed", "inspection_failed", "no_local_files"
        }
        for row in report
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()