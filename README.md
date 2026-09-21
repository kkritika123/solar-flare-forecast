# Solar Flare Forecast

A Python package for downloading and evaluating historical solar flare forecasts from the NASA CCMC Flare Scoreboard.

The package retrieves forecasts from multiple forecasting models, obtains observed solar flare events from the LMSAL SolarSoft archive, and evaluates forecast performance using the True Skill Statistic (TSS) and Heidke Skill Score (HSS).

The project currently supports historical evaluation for 2020–2025.

## Features

- Download historical forecasts from the NASA CCMC Flare Scoreboard
- Parse XML, TXT, and JSON forecast files
- Evaluate full-disk and active-region forecasts
- Automatically download LMSAL observed flare events when needed
- Calculate True Skill Statistic (TSS) and Heidke Skill Score (HSS)
- Evaluate C-, M-, and X-class flare thresholds
- Generate yearly and cumulative evaluation results
- Save processed forecasts and evaluation results as CSV files

## Installation

### Install from GitHub

Clone the repository and install the package:

```bash
git clone https://github.com/kkritika123/solar-flare-forecast.git
cd solar-flare-forecast
pip install .
```

Python 3.10 or later is required.

### Install from PyPI

After the package is published to PyPI, it can be installed with:

```bash
pip install solar-flare-forecast
```

## Quick Start

Download forecasts for a specific model and year:

```python
from flare_scoreboard import download_forecasts

download_forecasts(
    models=["NOAA_1"],
    years=[2024]
)
```

Evaluate the downloaded forecasts:

```python
from flare_scoreboard import evaluate

results = evaluate(
    models=["NOAA_1"],
    years=[2024]
)

print(results)
```

If the required LMSAL observation data is not available locally, `evaluate()` automatically downloads the observed flare events for the requested years.

A complete workflow can be run with:

```python
from flare_scoreboard import download_forecasts, evaluate

download_forecasts(
    models=["NOAA_1"],
    years=[2024]
)

results = evaluate(
    models=["NOAA_1"],
    years=[2024]
)

print(results)
```

## Download LMSAL Events Directly

Observed solar flare events can also be downloaded separately from the LMSAL SolarSoft archive:

```python
from flare_scoreboard import download_lmsal_events

events = download_lmsal_events(
    start="2024-01-01",
    end="2024-12-31",
    out="data/lmsal_events_2024.csv"
)

print(events.head())
```

This step is optional because `evaluate()` automatically downloads the required LMSAL event data when it is not already available.

## Repository Structure

```text
solar-flare-forecast/
│
├── flare_scoreboard/
│   ├── __init__.py          # Public package interface
│   ├── api.py               # Download and evaluation API
│   ├── config.py            # Configuration and built-in defaults
│   ├── constants.py         # CSV schema and constants
│   ├── crawl.py             # Discovers CCMC model folders
│   ├── csv_output.py        # Writes parsed forecast CSV files
│   ├── evaluation.py        # Event matching and TSS/HSS calculations
│   ├── http_client.py       # HTTP requests and downloads
│   ├── lmsal.py             # Downloads LMSAL observed flare events
│   ├── parse_core.py        # Time and probability helpers
│   ├── parsers.py           # XML, TXT, and JSON parsers
│   ├── pipeline.py          # Forecast processing pipeline
│   └── scoring.py           # Model evaluation and result generation
│
├── main.py
├── model.py
├── scrape_lmsal_events.py
├── plot_per_model_yearly_trends.py
├── config.json
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

The `flare_scoreboard` directory contains the installable Python package.
The root-level scripts are retained for the original research workflow and plotting utilities.

## System Pipeline

The diagram below shows how forecast data and observed flare events move through the system.

![Solar Flare Forecast System Pipeline](docs/images/system_pipeline.png)

`download_forecasts()` retrieves and parses forecast files from the CCMC archive. `evaluate()` matches the forecasts against observed LMSAL flare events and calculates TSS and HSS scores. If the required LMSAL event data is missing, it is downloaded automatically.

## Running the System

### Download Forecasts

Choose the models and years you want to process:

```python
from flare_scoreboard import download_forecasts

download_forecasts(
    models=["NOAA_1", "SIDC_v2"],
    years=[2024]
)
```

Parsed forecast files are saved in:

```text
output/<MODEL>/
```

### Evaluate Forecasts

After downloading the forecasts, run:

```python
from flare_scoreboard import evaluate

results = evaluate(
    models=["NOAA_1", "SIDC_v2"],
    years=[2024]
)

print(results)
```

Evaluation results are saved in:

```text
evaluation_results/<MODEL>/
```

The returned pandas DataFrame also contains the calculated TSS and HSS scores for further analysis.

### Plot Yearly Trends

The repository also includes a plotting script for generating TSS/HSS trend figures:

```bash
python plot_per_model_yearly_trends.py
```

To generate a combined grid:

```bash
python plot_per_model_yearly_trends.py --combined-grid --no-per-model --poster --dpi 300 --grid-ncols 3
```

## Evaluation

Forecasts are evaluated against observed solar flare events from the LMSAL SolarSoft archive.

The evaluation supports both:

- **Full-disk forecasts** — predictions for solar flare activity across the entire visible solar disk.
- **Active-region forecasts** — predictions associated with individual NOAA active regions.

Forecast performance is measured using the **True Skill Statistic (TSS)** and **Heidke Skill Score (HSS)**.

### True Skill Statistic (TSS)

TSS measures how well a forecasting system separates events from non-events:

```text
TSS = TP / (TP + FN) - FP / (FP + TN)
```

### Heidke Skill Score (HSS)

HSS measures forecast skill relative to random chance:

```text
HSS = 2(TP × TN - FN × FP) /
      ((TP + FN)(FN + TN) + (TP + FP)(FP + TN))
```

where:

- `TP` = True Positives
- `TN` = True Negatives
- `FP` = False Positives
- `FN` = False Negatives

The evaluation can calculate scores for C-, M-, and X-class flare thresholds depending on the forecast data available for each model.

## Models Evaluated

Ten CCMC models are currently included:

```
NOAA_1, SIDC_v2, ASSA_1, ASSA_24H_1, AMOS_v1,
ASAP_1, A-Effort, DAFFS, MagPy, SPS_1
```

## Data Sources

| Source | Description |
|---|---|
| [NASA CCMC Flare Scoreboard](https://ccmc.gsfc.nasa.gov/scoreboards/flare/) | Daily flare forecasts from multiple research groups (XML / JSON / TXT) |
| [LMSAL SolarSoft Latest Events](https://www.lmsal.com/solarsoft/latest_events_archive.html) | GOES flare event catalog used as ground truth |
