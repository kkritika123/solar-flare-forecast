
"""Command-line interface for Solar Flare Forecast."""

import sys


def main():
    """Run the Solar Flare Forecast command-line interface."""

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Solar Flare Forecast CLI")
        print()
        print("Usage:")
        print("  solarflarecast query [options]")
        print()
        print("Commands:")
        print("  query    Query HAPI or processed archive forecasts")
        return

    command = sys.argv[1]

    if command == "query":
        # Remove the subcommand so the existing parser
        # receives only its original arguments.
        sys.argv.pop(1)

        from query_forecasts import main as query_main
        query_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
