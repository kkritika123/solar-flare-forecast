# Flare Scoreboard Pipeline (Simple Guide)

Core workflow in this repo:
- Download + parse CCMC flare scoreboard forecasts
- Evaluate forecasts against LMSAL events (TSS/HSS)
- Plot yearly trends

## What you need

- Python 3.10+
- `config.json` set correctly (years/models)
- LMSAL CSV at `data/lmsal_events_2020_2025.csv` (or update `LMSAL_CSV` in `model.py`)

## Setup (once)

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main workflow

Run from project root:

```bash
python main.py
python model.py
python plot_per_model_yearly_trends.py
```

### If you want one figure with all models

```bash
python plot_per_model_yearly_trends.py --combined-grid --no-per-model --poster --dpi 300 --grid-ncols 3
```

## Important config fields (`config.json`)

- `years`: years to process
- `models`: model folder names to include (example: `["SPS_1"]`)
- `event_window_match_tolerance_hours`: matching tolerance around forecast window
- `forecast_window_fill_hours`: fill missing windows from issue time (usually `24`)

## Output folders

- Parsed forecasts: `output/<MODEL>/<YEAR>_full_disk.csv` and `output/<MODEL>/<YEAR>_region.csv`
- Scores: `evaluation_results/`
- Plots: `evaluation_results/figures/per_model_yearly_trends/`

## Optional: regenerate LMSAL events file

```bash
python scrape_lmsal_events.py --start 2020-01-01 --end 2025-12-31 --out data/lmsal_events_2020_2025.csv
```
