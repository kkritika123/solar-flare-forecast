# CCMC Flare Scoreboard: download + parse → yearly CSVs 
from flare_scoreboard import load_config, normalize_dir, discover_models, process_model


def _model_name_from_url(model_url: str) -> str:
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
    print("Models found:", len(models))
    for m in models:
        print(" -", _model_name_from_url(m))

    if models_filter is not None:
        # Respect config filter ("models" or "assa_format_models") if provided.
        if not models_filter:
            label = cfg.get("models_filter_label") or "models"
            print(
                f"\nConfig model list is empty ({label!r}). Add folder names, "
                "or remove that key to process all discovered models."
            )
            return
        discovered = {_model_name_from_url(u) for u in models}
        before = len(models)
        models = [u for u in models if _model_name_from_url(u) in models_filter]
        label = cfg.get("models_filter_label") or "models"
        print(
            f"\nFiltered by config {label!r}: {len(models)} of {before} "
            f"folder(s) will be processed."
        )
        not_found = sorted(models_filter - discovered)
        if not_found:
            print("  [WARN] These names were not found on the server:", ", ".join(not_found))

    # Process each selected model end-to-end (download + parse + yearly CSV write).
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

    print("\nDone. Output is in output/<MODEL>/<YEAR>_full_disk.csv and <YEAR>_region.csv")


if __name__ == "__main__":
    main()
