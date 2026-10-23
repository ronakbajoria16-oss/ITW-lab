# SolarScope · Group 8

Renewable energy generation analysis and solar power forecasting, with the original green-and-gold dashboard and a new cinematic solar-field opening inspired by the supplied visual reference. The solar-field bitmap is original generated artwork; no OpenAI branding, product claims, or site assets are embedded.

The current stylesheet integrates the user's supplied Inter typography, centered cosmic hero, dark navigation bar, teal gradient secondary hero, gold accents, and sidebar/footer styling. Streamlit-specific font overrides and responsive rules adapt that design to the existing application. Inter loads through Google Fonts when available, with a sans-serif fallback offline.

## Run locally

Use Python 3.11 or 3.12 for a new installation. In this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux activate with `source .venv/bin/activate`. The Windows launcher also starts the app once the intended Python environment and dependencies are available. The `.streamlit/config.toml` theme is loaded when starting from this folder.

## What is implemented

- **Sources:** synthetic demo, hourly/15-minute CSV, original Kaggle plant-generation/weather CSV-pair importer, NASA POWER historical hourly API for locations including Ranchi.
- **Cleaning:** timestamp parsing, numeric/range validation, fixed-inverter-fleet aggregation, complete-hour averaging, explicit missing targets, short causal weather forward fill, cleaning audit summary.
- **EDA:** daily/monthly energy, daylight filter, irradiance/AC/DC/weather correlation heatmap, temperature/irradiance-normalized output, hourly peak, cloud associations when cloud data exists.
- **Features:** hour, month, day of year, cyclical time, previous-hour and previous-day power, shifted 3-hour/24-hour rolling power.
- **Models:** Linear Regression, Random Forest, XGBoost, TensorFlow Dense, and a true 24-hour TensorFlow LSTM.
- **Evaluation:** chronological 70/15/15 train/validation/test, training-only fitted preprocessing, validation-based selection, MAE/RMSE/R², previous-day baseline, and actual/predicted plot.
- **Forecast:** recursive 24/48/72/168-hour generation from supplied future weather or an explicitly labeled historical weather scenario; manual next-hour weather inputs.
- **State capacity:** validated CSV analysis with report dates, source URLs, and separate solar/wind/other-renewable MW categories. MNRE/CEA report links are provided; PDF scraping is not automated.
- **Exports:** cleaned CSV, features, metrics, held-out predictions, forecasts, trained models/preprocessing, and a generated Markdown project report with all ten requested sections.

## First use

Select **Demo plant → Train & evaluate → Train selected models**. Then inspect **Forecast**, or open **Try your own weather** to predict the next hour. Try the neural models separately if you want faster classical comparisons first. A long horizon or LSTM may take longer on a CPU.

No performance threshold is guaranteed. The synthetic demo is intentionally simple and cannot validate real-plant accuracy. Correlation >0.9 and R² ≥0.95 are hypotheses to check, not prefilled results.

## CSV contract

Required history: `timestamp,irradiance_wm2,temperature_c,power_kw`.
Optional weather: `humidity_pct,cloud_cover_pct,module_temperature_c`.
Optional descriptive DC: `dc_power_kw` (excluded from prediction features to avoid using same-hour output).

- One plant, unique timestamps on hourly or quarter-hour boundaries; at least 240 complete hourly power observations after cleaning.
- Hourly or quarter-hour readings must represent interval-mean kW. Complete 15-minute groups are averaged to hourly mean power; four instantaneous readings do not establish interval energy unless representative.
- All timestamps must be timezone-free and use one documented time standard. Never combine plant-local timestamps with UTC silently. Resolve daylight-saving transitions before import.
- Radiation in W/m², temperature in °C, humidity/cloud cover in 0–100%. Negative power/radiation and invalid percentages become missing. Infinite values become missing.
- Missing power is never interpolated or replaced with zero. Incomplete hours remain missing. Weather may be forward-filled for at most two hours; any remaining missing predictor values are handled with training-fitted median imputation.
- Feature construction uses a continuous hourly grid. Rows without complete lag/rolling history are excluded from training/evaluation. Forecasts require complete measured power for the final 48 hours.
- Night-time rows remain in energy accounting and recursive forecasts. Daylight-only fitting/EDA are separate options.
- Daily/monthly energy excludes missing hours; the coverage card and audit summary expose gaps. Partial days/months are included.

Future weather requires `timestamp` plus the weather columns used by the trained model, starting one hour after history with no gaps. If module temperature was available during training, supply a future estimate of it as well. Forecasts are nonnegative and zero when irradiance is zero; no plant-capacity ceiling or physical uncertainty bounds are claimed.

## Kaggle

Dataset reference: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Obtain the original `Plant_1_Generation_Data.csv` / `Plant_1_Weather_Sensor_Data.csv` pair, or Plant 2's pair. Select **Kaggle plant**, upload both, and select the plant ID. The public download endpoint returned HTTP 404 during this implementation, so real Kaggle data is **not bundled** and no Kaggle accuracy results are claimed. The importer was tested with a fixture matching the original schema.

The importer parses Plant 1 DD-MM-YYYY and Plant 2 ISO dates, joins on plant and time, sums AC/DC across a fixed fleet, and invalidates timestamps with a missing inverter. Irradiation is converted from kW/m² to W/m². Some source DC scales require investigation: the explicit **DC unit correction divisor** defaults to 1 and is recorded in the audit. It never alters AC targets. Humidity and cloud cover are not supplied by this dataset and are not invented.

## NASA POWER

The free API requests `ALLSKY_SFC_SW_DWN,T2M,RH2M` with renewable-energy community and **UTC**. Hourly Wh/m² over one hour is numerically equal to interval-mean W/m²; unexpected units are rejected. NASA fill values become missing. Location/range/network/API errors are displayed in the app. Fetch is explicit, never automatic on every interaction.

NASA provides historical gridded weather, not plant generation or future weather forecasts. Download weather without training, or upload matching measured power in UTC. Convert IST to UTC by subtracting 5h30m and aggregate into the same UTC hourly bins. The default Ranchi API retrieval was verified with 336 hours for 2025-01-01 through 2025-01-14. The included `sample_data/NASA_Ranchi_UTC.csv` is that weather-only sample; it is not used as a fabricated training target.

Docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/

## Capacity analysis

Use **State capacity** to download the empty template. Copy verified figures from one dated MNRE/CEA report. CSV fields: `state,solar_mw,wind_mw,other_re_mw,as_of,source_url`. Use one row per state and one date, with HTTPS source URLs. Define non-overlapping categories for `other_re_mw`; do not add report totals to their constituent figures. Capacity is MW, not energy or forecast target data.

## Evaluation limits

Selection uses validation RMSE. Test scores represent rolling **one-hour-ahead** predictions with observed current weather and earlier measured power. Multi-hour forecasts recursively substitute predicted power for unavailable lags. Therefore test scores do not establish operational day-ahead accuracy. A real operational backtest needs weather forecasts available at issuance time, multiple origins/seasons, and uncertainty evaluation. The app does not claim causal temperature effects or physical panel efficiency; panel area and plane-of-array irradiance are unavailable.

## Saved-model inference

Export a model in **Data & exports**, extract its ZIP, then:

```powershell
python predict.py --model model-folder --history cleaned_history.csv --weather future_weather.csv --output forecast.csv
```

Only load trusted model archives: joblib/pickle model files can execute code. Archives contain preprocessing and feature order. Neural bundles include a Keras model; classical bundles include their fitted pipeline. Keep compatible library versions for reuse. `core.py` retains the original TensorFlow-only workflow for backwards compatibility; the expanded dashboard uses `research.py`.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests exercise causal features, missing-target preservation, incomplete inverter/hour aggregation, NASA units/fill values, model comparisons, recursive forecasts, model reload, true LSTM training, capacity validation, report structure, and the Streamlit training/forecast workflow.

## Future scope

IoT sensors, live external weather forecasts, battery dispatch, joint solar/wind prediction, uncertainty intervals, and rolling-origin operational backtests are roadmap items. They are documented in the generated report and are not presented as completed integrations.
