"""Download CCMC flare forecasts and write yearly CSVs per model.

Settings come from config.json. Output goes to output/<MODEL>/<YEAR>_full_disk.csv
and <YEAR>_region.csv. Run this first; then run model.py to score the results.
"""
from flare_scoreboard import load_config, normalize_dir, discover_models, process_model


def _model_name(model_url: str) -> str:
    return model_url.rstrip("/").split("/")[-1]


def main():
    cfg = load_config()

    base_url = normalize_dir(cfg["base_url"])
    years_set = cfg["years_set"]
    parse_exts = cfg["parse_exts_set"]
    raw_dir = cfg["raw_dir"]
    out_dir = cfg["out_dir"]
    workers = cfg["workers"]
    download_all = cfg["download_all_files"]
    models_filter = cfg.get("models_filter")

    print("Discovering model folders...")
    models = discover_models(base_url)
    print(f"Models found: {len(models)}")
    for m in models:
        print(f" - {_model_name(m)}")

    if models_filter is not None:
        label = cfg.get("models_filter_label") or "models"
        if not models_filter:
            print(f"\nConfig model list {label!r} is empty. Add names or remove the key.")
            return

        discovered = {_model_name(u) for u in models}
        before = len(models)
        models = [u for u in models if _model_name(u) in models_filter]
        print(f"\nFiltered by {label!r}: {len(models)} of {before} folder(s) will be processed.")

        not_found = sorted(models_filter - discovered)
        if not_found:
            print("  [WARN] Not found on server:", ", ".join(not_found))

    for model_url in models:
        process_model(
            model_url=model_url,
            years_set=years_set,
            parse_exts=parse_exts,
            raw_dir=raw_dir,
            out_dir=out_dir,
            workers=workers,
            download_all=download_all,
        )

    print("\nDone. Outputs are in output/<MODEL>/<YEAR>_full_disk.csv and <YEAR>_region.csv")


if __name__ == "__main__":
    main()
