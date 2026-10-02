import argparse
from pathlib import Path

from flare_scoreboard import compare_sources


def main():
    parser = argparse.ArgumentParser(
        description="Compare HAPI and processed archive forecasts."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--type",
        choices=["full_disk", "active_region"],
        required=True,
    )
    parser.add_argument("--flare", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--archive-dir", default="output")
    parser.add_argument("--out", default="comparison.csv")

    args = parser.parse_args()

    summary, details = compare_sources(
        model_id=args.model,
        model_type=args.type,
        flare_type=args.flare,
        start=args.start,
        end=args.end,
        archive_dir=args.archive_dir,
    )

    for name, value in summary.items():
        print(f"{name}: {value}")

    folder = Path("data") / "comparison"
    folder.mkdir(parents=True, exist_ok=True)

    output = folder / Path(args.out).name
    details.to_csv(output, index=False)
    print(f"Comparison saved to {output}")


if __name__ == "__main__":
    main()