# Solar Flare Forecast Verification

A research project where I compare solar flare forecasts from the NASA CCMC
Flare Scoreboard against observed flare events from the LMSAL SolarSoft archive.
The pipeline downloads every forecast each model published between 2020 and
2025, scores them all under the same rules (TSS and HSS), and plots yearly
trends so I can see, year by year, which model performed best.

## Repository Structure

```
solar-flare-forecast/
│
├── main.py                          # Step 1: download + parse CCMC forecasts
├── model.py                         # Step 2: score forecasts vs LMSAL events
├── plot_per_model_yearly_trends.py  # Step 3: plot yearly TSS/HSS trends
├── scrape_lmsal_events.py           # Builds the LMSAL events CSV
├── flare_eval_utils.py              # Matching + TSS/HSS helpers
│
├── flare_scoreboard/                # Download + parsing package
│   ├── config.py                    #   loads config.json
│   ├── http_client.py               #   HTTP session, listing, downloads
│   ├── crawl.py                     #   discovers model folders on CCMC
│   ├── parsers.py                   #   XML / TXT / JSON parsers
│   ├── parse_core.py                #   small time / probability helpers
│   ├── csv_output.py                #   writes yearly CSVs per model
│   ├── pipeline.py                  #   one-model orchestration
│   └── constants.py                 #   CSV schema + network defaults
│
├── tests/                           # Parser sanity tests
├── config.json                      # Years, models, matching settings
├── ccmc_models_overview.csv         # Model descriptions + papers
├── requirements.txt
├── .gitignore
└── README.md
```

## System Pipeline

How the files work and connect to each other. `main.py` downloads the raw
forecasts from CCMC and writes a clean CSV per model and year. `model.py`
reads those CSVs, matches each forecast to LMSAL events, and writes TSS/HSS
tables. `plot_per_model_yearly_trends.py` turns those tables into PNG plots.
`scrape_lmsal_events.py` is only needed once — it builds the LMSAL events CSV
that `model.py` uses as ground truth.

```
  CCMC Flare Scoreboard            LMSAL SolarSoft Archive
           │                                │
           ▼                                ▼
        main.py                   scrape_lmsal_events.py
   download + parse                 build events CSV
           │                                │
           ▼                                ▼
  output/<MODEL>/*.csv        data/lmsal_events_*.csv
               \                   /
                \                 /
                 ▼               ▼
                      model.py
               match + score (TSS/HSS)
                         │
                         ▼
           evaluation_results/<MODEL>/*.csv
                         │
                         ▼
         plot_per_model_yearly_trends.py
                         │
                         ▼
        evaluation_results/figures/*.png
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/kkritika123/solar-flare-forecast.git
cd solar-flare-forecast
```

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Build the LMSAL events file (one time)

`model.py` needs a CSV of observed flare events. If `data/lmsal_events_2020_2025.csv`
does not exist yet, build it:

```bash
python scrape_lmsal_events.py --start 2020-01-01 --end 2025-12-31 --out data/lmsal_events_2020_2025.csv
```

### 4. Configure (optional)

Edit `config.json` to change the years processed or limit which models to run:

```json
{
  "years": [2020, 2021, 2022, 2023, 2024, 2025],
  "models": [],
  "event_window_match_tolerance_hours": 2,
  "forecast_window_fill_hours": 24
}
```

Leave `"models": []` to process every model available on the scoreboard. Or
list specific ones, e.g. `["SPS_1", "NOAA_1"]`.

## Running the System

### Download and parse all forecasts

```bash
python main.py
```

Writes `output/<MODEL>/<YEAR>_full_disk.csv` and `<YEAR>_region.csv` per model.

### Score forecasts against LMSAL events

```bash
python model.py
```

Writes per-model tables in `evaluation_results/<MODEL>/` and combined tables
in `evaluation_results/all_models_*.csv`.

### Plot yearly trends

One figure per model (full-disk and region separately, the default):

```bash
python plot_per_model_yearly_trends.py
```

One combined figure with every model in a grid (good for posters / slides):

```bash
python plot_per_model_yearly_trends.py --combined-grid --no-per-model --poster --dpi 300 --grid-ncols 3
```

Plot just one forecast type:

```bash
python plot_per_model_yearly_trends.py --forecast-type full_disk
python plot_per_model_yearly_trends.py --forecast-type region
```

## Evaluation

`model.py` computes TSS and HSS per year, per forecast type (full-disk / region),
and per flare class threshold (C / M / X). Example row from
`evaluation_results/all_models_yearly_scores.csv`:

```
model_name   year  forecast_type  threshold  n_forecasts   TSS     HSS
-----------  ----  -------------  ---------  -----------  ------  ------
NOAA_1       2023  full_disk      M                 365   +0.42   +0.31
SIDC_v2      2023  full_disk      M                 365   +0.38   +0.29
ASSA_1       2023  full_disk      M                 365   +0.46   +0.33
```

Metric definitions (probability threshold = 0.5):

- **TSS** (True Skill Statistic) = POD − POFD. Range [−1, 1]; 0 = no skill.
- **HSS** (Heidke Skill Score) = skill relative to random chance. Range (−∞, 1]; 1 = perfect.

A forecast is a true positive if at least one LMSAL flare of that class or
higher starts inside the forecast window (with a small tolerance). Region
forecasts additionally require the NOAA AR number to match.

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
