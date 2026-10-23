"""Data ingestion, causal features, model comparison, and report generation."""
from datetime import date
import io
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KAGGLE_URL = 'https://www.kaggle.com/datasets/anikannal/solar-power-generation-data'
NASA_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
WEATHER_NAMES = ['irradiance_wm2', 'temperature_c', 'humidity_pct', 'cloud_cover_pct', 'module_temperature_c']


def parse_dates(series):
    result = pd.to_datetime(series, format='mixed', dayfirst=False, errors='coerce')
    # Kaggle Plant 1 uses DD-MM-YYYY, Plant 2 uses ISO timestamps.
    if series.astype(str).str.match(r'^\d{2}-\d{2}-\d{4}').any():
        result = pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')
    if result.isna().any():
        raise ValueError('Invalid timestamp. Use ISO dates or the original Kaggle DATE_TIME format.')
    if result.dt.tz is not None:
        raise ValueError('Use timezone-free timestamps in a single, documented time standard.')
    return result


def clean_history(frame):
    """Hourly means; causal short weather fill; never fill measured targets."""
    df = frame.copy()
    required = ['timestamp', 'irradiance_wm2', 'temperature_c', 'power_kw']
    if not set(required).issubset(df):
        raise ValueError('Required columns: ' + ', '.join(required))
    df['timestamp'] = parse_dates(df.timestamp)
    if df.timestamp.duplicated().any():
        raise ValueError('Duplicate timestamps: aggregate inverter readings with the Kaggle importer first.')
    if not df.timestamp.dt.minute.isin([0, 15, 30, 45]).all() or df.timestamp.dt.second.ne(0).any():
        raise ValueError('Use hourly or 15-minute timestamps on quarter-hour boundaries.')
    numeric = [c for c in WEATHER_NAMES + ['power_kw', 'dc_power_kw'] if c in df]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in ['irradiance_wm2', 'power_kw', 'dc_power_kw']:
        if c in df:
            df.loc[df[c] < 0, c] = np.nan
    for c in ['humidity_pct', 'cloud_cover_pct']:
        if c in df:
            df.loc[~df[c].between(0, 100), c] = np.nan
    df = df.sort_values('timestamp').set_index('timestamp')[numeric]
    step = df.index.to_series().diff().dropna().min()
    if step not in [pd.Timedelta(minutes=15), pd.Timedelta(hours=1)]:
        raise ValueError('History must contain hourly or 15-minute observations.')
    expected = 4 if step == pd.Timedelta(minutes=15) else 1
    counts = df.resample('h').count()
    hourly = df.resample('h').mean()
    # An average from a partial hour must not be presented as complete energy.
    hourly = hourly.where(counts >= expected)
    weather = [c for c in WEATHER_NAMES if c in hourly]
    missing_before = int(hourly[weather].isna().sum().sum())
    hourly[weather] = hourly[weather].ffill(limit=2)
    missing_after = int(hourly[weather].isna().sum().sum())
    report = dict(input_rows=len(frame), hourly_rows=len(hourly),
                  missing_power_hours=int(hourly.power_kw.isna().sum()),
                  weather_values_forward_filled=missing_before-missing_after,
                  remaining_weather_gaps=missing_after, interval_minutes=15 if expected == 4 else 60)
    if hourly.power_kw.notna().sum() < 240:
        raise ValueError('At least 240 complete hourly power observations are required after cleaning.')
    return hourly.reset_index(), report


def import_kaggle(generation, weather, plant_id=None, dc_divisor=1.):
    gen, met = generation.copy(), weather.copy()
    required_gen = {'DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER'}
    required_met = {'DATE_TIME', 'PLANT_ID', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION'}
    if not required_gen.issubset(gen) or not required_met.issubset(met):
        raise ValueError('Upload the original Kaggle generation and weather CSV pair.')
    gen['PLANT_ID'], met['PLANT_ID'] = gen.PLANT_ID.astype(str), met.PLANT_ID.astype(str)
    plants = sorted(set(gen.PLANT_ID) & set(met.PLANT_ID))
    if plant_id is None and len(plants) == 1:
        plant_id = plants[0]
    if str(plant_id) not in plants:
        raise ValueError('Select one plant present in both files.')
    gen, met = gen[gen.PLANT_ID == str(plant_id)].copy(), met[met.PLANT_ID == str(plant_id)].copy()
    gen['timestamp'], met['timestamp'] = parse_dates(gen.DATE_TIME), parse_dates(met.DATE_TIME)
    if gen.duplicated(['timestamp', 'SOURCE_KEY']).any() or met.timestamp.duplicated().any():
        raise ValueError('Duplicate inverter readings or weather timestamps found.')
    for col in ['AC_POWER', 'DC_POWER']:
        gen[col] = pd.to_numeric(gen[col], errors='coerce')
        gen.loc[gen[col] < 0, col] = np.nan
    total_inverters = gen.SOURCE_KEY.nunique()
    grouped = gen.groupby('timestamp')
    power = grouped[['AC_POWER', 'DC_POWER']].sum(min_count=total_inverters)
    incomplete = grouped.SOURCE_KEY.nunique().ne(total_inverters)
    power.loc[incomplete, :] = np.nan
    if dc_divisor <= 0:
        raise ValueError('DC divisor must be positive.')
    power = power.rename(columns={'AC_POWER': 'power_kw', 'DC_POWER': 'dc_power_kw'})
    power['dc_power_kw'] /= dc_divisor
    met = met.rename(columns={'AMBIENT_TEMPERATURE': 'temperature_c', 'MODULE_TEMPERATURE': 'module_temperature_c', 'IRRADIATION': 'irradiance_wm2'})
    met['irradiance_wm2'] = pd.to_numeric(met.irradiance_wm2, errors='coerce') * 1000
    merged = power.join(met.set_index('timestamp')[['temperature_c', 'module_temperature_c', 'irradiance_wm2']], how='outer').reset_index()
    result, report = clean_history(merged)
    report.update(plant_id=str(plant_id), inverter_count=total_inverters,
                  incomplete_inverter_intervals=int(incomplete.sum()), dc_divisor=float(dc_divisor))
    return result, report


def parse_nasa(payload):
    parameters = payload.get('properties', {}).get('parameter', {})
    mapping = {'ALLSKY_SFC_SW_DWN': 'irradiance_wm2', 'T2M': 'temperature_c', 'RH2M': 'humidity_pct'}
    if not all(k in parameters for k in mapping):
        raise ValueError('NASA response is missing requested parameters.')
    radiation_unit = payload.get('parameters', {}).get('ALLSKY_SFC_SW_DWN', {}).get('units', '').lower().replace(' ', '')
    if radiation_unit not in ['wh/m^2', 'wh/m2', 'w/m^2', 'w/m2']:
        raise ValueError(f'Unexpected NASA hourly irradiance unit: {radiation_unit!r}. Conversion requires review.')
    df = pd.DataFrame({target: parameters[source] for source, target in mapping.items()})
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H', errors='raise')
    df.index.name = 'timestamp'
    fill = payload.get('header', {}).get('fill_value', -999)
    df = df.apply(pd.to_numeric, errors='coerce').astype(float)
    df = df.mask(df.isin({fill, -999}))
    # Hourly Wh/m² over a one-hour interval is numerically equal to mean W/m².
    return df.sort_index().reset_index()


def fetch_nasa(latitude, longitude, start, end):
    import requests
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('Invalid latitude or longitude.')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or end-start > pd.Timedelta(days=366):
        raise ValueError('Choose a date range of at most 366 days, with start before end.')
    if start.year < 2001 or end.date() >= date.today():
        raise ValueError('NASA hourly history starts in 2001. Choose an end date before today.')
    response = requests.get(NASA_URL, params=dict(parameters='ALLSKY_SFC_SW_DWN,T2M,RH2M', community='RE',
                            latitude=latitude, longitude=longitude, start=start.strftime('%Y%m%d'),
                            end=end.strftime('%Y%m%d'), format='JSON', **{'time-standard': 'UTC'}), timeout=(10, 90))
    response.raise_for_status()
    return parse_nasa(response.json())


def causal_features(df, weather_columns=None):
    weather_columns = weather_columns or [c for c in WEATHER_NAMES if c in df and df[c].notna().any()]
    x = df[weather_columns].copy()
    hour, day = df.timestamp.dt.hour, df.timestamp.dt.dayofyear
    x['hour'], x['month'], x['day_of_year'] = hour, df.timestamp.dt.month, day
    x['hour_sin'], x['hour_cos'] = np.sin(hour*2*np.pi/24), np.cos(hour*2*np.pi/24)
    x['year_sin'], x['year_cos'] = np.sin(day*2*np.pi/365.25), np.cos(day*2*np.pi/365.25)
    # df is on a complete hourly grid. Every lag excludes current/future target values.
    x['power_lag_1h'] = df.power_kw.shift(1)
    x['power_lag_24h'] = df.power_kw.shift(24)
    x['power_rolling_3h'] = df.power_kw.shift(1).rolling(3, min_periods=3).mean()
    x['power_rolling_24h'] = df.power_kw.shift(1).rolling(24, min_periods=24).mean()
    return x


def scores(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if len(y) < 2:
        return {'MAE (kW)': np.nan, 'RMSE (kW)': np.nan, 'R²': np.nan, 'Hours': len(y)}
    return {'MAE (kW)': float(mean_absolute_error(y, pred)), 'RMSE (kW)': float(np.sqrt(mean_squared_error(y, pred))),
            'R²': float(r2_score(y, pred)) if np.var(y) > 0 else np.nan, 'Hours': len(y)}


def clip_power(pred, weather):
    result = np.maximum(0, np.asarray(pred).ravel())
    result[weather.irradiance_wm2.to_numpy() <= 0] = 0
    return result


def fit_models(df, names, daylight=False, epochs=20):
    weather_cols = [c for c in WEATHER_NAMES if c in df and df[c].notna().any()]
    x = causal_features(df, weather_cols)
    valid = df.power_kw.notna() & df.irradiance_wm2.notna() & x[['power_lag_1h', 'power_lag_24h', 'power_rolling_3h', 'power_rolling_24h']].notna().all(axis=1)
    if daylight:
        valid &= df.irradiance_wm2 > 10
    indices = np.flatnonzero(valid)
    if len(indices) < 120:
        raise ValueError('Not enough usable observations after causal feature construction. Need at least 120; check gaps or disable daylight filtering.')
    a, b = int(len(indices)*.7), int(len(indices)*.85)
    training, validation, testing = indices[:a], indices[a:b], indices[b:]
    # All models have the same target rows and use only earlier observations for fitting.
    y = df.power_kw
    output = df.iloc[testing][['timestamp', 'irradiance_wm2', 'power_kw']].copy()
    output['Previous day'] = y.shift(24).iloc[testing].to_numpy()
    bundles, rows, losses = {}, [], {}
    constructors = {'Linear Regression': lambda: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LinearRegression()),
                    'Random Forest': lambda: make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=2))}
    for name in names:
        if name == 'XGBoost':
            from xgboost import XGBRegressor
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), XGBRegressor(n_estimators=220, max_depth=4, learning_rate=.05, subsample=.9, colsample_bytree=.9, objective='reg:squarederror', random_state=42, n_jobs=2))
        elif name in constructors:
            model = constructors[name]()
        elif name in ['TensorFlow Dense', 'TensorFlow LSTM']:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(42)
            preprocess = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
            z = preprocess.fit_transform(x.iloc[training]).astype('float32')
            zv, zt = preprocess.transform(x.iloc[validation]).astype('float32'), preprocess.transform(x.iloc[testing]).astype('float32')
            target_scale = max(float(y.iloc[training].max()), 1.)
            if name == 'TensorFlow LSTM':
                # A true sequence of the preceding 24 hourly feature rows, ending at the target hour.
                all_z = preprocess.transform(x).astype('float32')
                sequence = lambda ids: np.stack([all_z[i-23:i+1] for i in ids])
                z, zv, zt = sequence(training), sequence(validation), sequence(testing)
                layers = [tf.keras.layers.Input((24, x.shape[1])), tf.keras.layers.LSTM(32), tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(1)]
            else:
                layers = [tf.keras.layers.Input((x.shape[1],)), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1)]
            model = tf.keras.Sequential(layers)
            model.compile(optimizer='adam', loss='mse')
            history = model.fit(z, y.iloc[training].to_numpy()/target_scale, validation_data=(zv, y.iloc[validation].to_numpy()/target_scale), epochs=epochs, batch_size=64, shuffle=False, verbose=0,
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            losses[name] = pd.DataFrame(history.history)
            prediction = clip_power(np.asarray(model(zt, training=False)).ravel()*target_scale, df.iloc[testing])
            val_prediction = clip_power(np.asarray(model(zv, training=False)).ravel()*target_scale, df.iloc[validation])
            bundles[name] = dict(model=model, preprocess=preprocess, scale=target_scale, kind=name, columns=list(x), weather_columns=weather_cols)
            output[name] = prediction
            rows.append(dict(Model=name, **scores(y.iloc[testing], prediction), **{'Validation RMSE (kW)': scores(y.iloc[validation], val_prediction)['RMSE (kW)']}))
            continue
        else:
            raise ValueError('Unknown model: ' + name)
        model.fit(x.iloc[training], y.iloc[training])
        prediction = clip_power(model.predict(x.iloc[testing]), df.iloc[testing])
        val_prediction = clip_power(model.predict(x.iloc[validation]), df.iloc[validation])
        bundles[name] = dict(model=model, kind='sklearn', columns=list(x), weather_columns=weather_cols)
        output[name] = prediction
        rows.append(dict(Model=name, **scores(y.iloc[testing], prediction), **{'Validation RMSE (kW)': scores(y.iloc[validation], val_prediction)['RMSE (kW)']}))
    if not rows:
        raise ValueError('Choose at least one model.')
    table = pd.DataFrame(rows).sort_values('Validation RMSE (kW)').reset_index(drop=True)
    table = pd.concat([table, pd.DataFrame([dict(Model='Previous day', **scores(output.power_kw, output['Previous day']), **{'Validation RMSE (kW)': np.nan})])], ignore_index=True)
    return dict(bundles=bundles, predictions=output, metrics=table, losses=losses, best=table.Model.iloc[0],
                splits=dict(train=len(training), validation=len(validation), test=len(testing)), daylight=daylight,
                split_dates=dict(train_end=str(df.timestamp.iloc[training[-1]]), validation_end=str(df.timestamp.iloc[validation[-1]]), test_start=str(df.timestamp.iloc[testing[0]])))


def predict_row(bundle, frame):
    frame = frame[bundle['columns']]
    if bundle['kind'] == 'sklearn':
        return float(bundle['model'].predict(frame.tail(1))[0])
    z = bundle['preprocess'].transform(frame).astype('float32')
    if bundle['kind'] == 'TensorFlow LSTM':
        z = z[-24:].reshape(1, 24, z.shape[1])
    else:
        z = z[-1:]
    return float(np.asarray(bundle['model'](z, training=False)).ravel()[0]*bundle['scale'])


def forecast(bundle, history, future):
    needed = bundle['weather_columns']
    if not set(['timestamp']+needed).issubset(future):
        raise ValueError('Future weather requires: timestamp, ' + ', '.join(needed))
    future = future[['timestamp']+needed].copy()
    future['timestamp'] = parse_dates(future.timestamp)
    future = future.sort_values('timestamp').reset_index(drop=True)
    expected = pd.date_range(history.timestamp.iloc[-1]+pd.Timedelta(hours=1), periods=len(future), freq='h')
    if future.empty or not np.array_equal(future.timestamp.to_numpy(), expected.to_numpy()):
        raise ValueError('Future weather must be contiguous hourly rows starting one hour after history.')
    for c in needed:
        future[c] = pd.to_numeric(future[c], errors='coerce')
        if not np.isfinite(future[c]).all():
            raise ValueError(f'Future {c} values must be finite.')
        if c in ['humidity_pct', 'cloud_cover_pct'] and not future[c].between(0, 100).all():
            raise ValueError(f'{c} must be between 0 and 100.')
    if (future.irradiance_wm2 < 0).any():
        raise ValueError('Irradiance cannot be negative.')
    if history.tail(48).power_kw.isna().any():
        raise ValueError('Forecast requires complete measured power in the last 48 hours. Resolve recent gaps first.')
    rolling_history = history.copy()
    predictions = []
    for _, row in future.iterrows():
        added = row.to_frame().T
        added['timestamp'] = pd.to_datetime(added.timestamp)
        added['power_kw'] = np.nan
        combined = pd.concat([rolling_history, added], ignore_index=True)
        x = causal_features(combined, needed).tail(24)
        value = max(0., predict_row(bundle, x)) if row.irradiance_wm2 > 0 else 0.
        predictions.append(value)
        combined.loc[combined.index[-1], 'power_kw'] = value
        rolling_history = combined
    future['predicted_power_kw'] = predictions
    return future


def future_scenario(df, hours):
    cols = [c for c in WEATHER_NAMES if c in df and df[c].notna().any()]
    profile = df.tail(168).groupby(df.tail(168).timestamp.dt.hour)[cols].mean()
    dates = pd.date_range(df.timestamp.iloc[-1]+pd.Timedelta(hours=1), periods=hours, freq='h')
    future = profile.reindex(dates.hour).reset_index(drop=True)
    future.insert(0, 'timestamp', dates)
    return future


def validate_capacity(df):
    required = ['state', 'solar_mw', 'wind_mw', 'other_re_mw', 'as_of', 'source_url']
    if not set(required).issubset(df):
        raise ValueError('Capacity CSV needs: ' + ', '.join(required))
    out = df[required].copy()
    if out[required].isna().any().any() or out.state.astype(str).str.strip().eq('').any():
        raise ValueError('Capacity rows must have a state, figures, date, and source URL.')
    out['as_of'] = pd.to_datetime(out.as_of, errors='coerce')
    if out.as_of.isna().any() or out.as_of.nunique() != 1 or out.state.duplicated().any():
        raise ValueError('Use one report date and one row per state.')
    for c in ['solar_mw', 'wind_mw', 'other_re_mw']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
        if not np.isfinite(out[c]).all() or (out[c] < 0).any():
            raise ValueError('Capacity figures must be finite nonnegative MW.')
    if not out.source_url.str.match(r'^https://').all():
        raise ValueError('Use HTTPS source URLs for the capacity report.')
    out['total_re_mw'] = out[['solar_mw', 'wind_mw', 'other_re_mw']].sum(axis=1)
    return out.sort_values('total_re_mw', ascending=False)


def report_markdown(df, result=None, source='Synthetic demo', cleaning=None):
    daylight = df[df.irradiance_wm2 > 10]
    corr = daylight[['irradiance_wm2', 'power_kw']].corr().iloc[0, 1]
    peak = df.groupby(df.timestamp.dt.hour).power_kw.mean().idxmax()
    metrics_text = 'Models have not yet been trained. No accuracy claims can be made.'
    if result:
        table = result['metrics']
        metrics_text = table.to_csv(index=False) + '\nSelected by validation RMSE: ' + result['best']
    return f'''# SolarScope — Renewable Energy Generation Analysis and Solar Power Forecasting

## Abstract
This implementation analyzes historical plant generation and weather, compares predictive models, and exposes weather-conditioned forecasting through a Streamlit dashboard. Source: {source}. Report values below are computed from the loaded data.

## Introduction
Solar generation varies with irradiance, temperature, cloud cover, and plant operating conditions. Forecasting can support planning, while data quality and weather forecast uncertainty limit operational accuracy.

## Literature Review
The project follows chronological time-series evaluation and training-only preprocessing. Linear regression provides an interpretable baseline; random forests model nonlinear interactions; XGBoost uses gradient-boosted trees. Optional TensorFlow dense and LSTM models provide neural comparisons. NASA POWER supplies historical gridded meteorological observations, not future weather forecasts. References below document these methods and data sources; this is a technical overview, not a systematic literature review.

## Problem Statement
Estimate hourly AC power (kW) from weather and prior generation, without using future targets during fitting or feature construction. Separate state-level installed capacity (MW) from plant-level energy (kWh).

## Methodology
Raw CSV/API data → validation → complete hourly aggregation → causal weather forward fill (maximum two hours) → EDA → hour/month/day-of-year and lag/rolling features → chronological 70/15/15 split → training → validation-based model selection → untouched test evaluation → recursive forecast.
Missing power is never imputed. Partial hours and incomplete inverter totals are excluded from valid measurements. Rolling averages shift by one hour. Optional daylight-only fitting excludes irradiance ≤10 W/m²; energy totals retain all observed hours. Test evaluation is rolling one-hour-ahead with observed weather and observed prior power. Future forecasts recursively use predicted power. These protocols have different uncertainty.

## Implementation
Python, Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras, Altair, and Streamlit. The original green-and-gold SolarScope workspace remains the dashboard foundation. Kaggle importer joins generation and weather by timestamp for one selected plant, sums inverter power, averages complete quarter-hour observations into hourly values, and converts irradiation from kW/m² to W/m². DC scale is user-confirmed and affects descriptive analysis only. NASA requests UTC hourly history. Capacity data requires a report date and source URL.

## Results and Discussion
History: {len(df):,} hourly bins. Observed energy: {df.power_kw.sum():,.2f} kWh (missing hours excluded). Daylight irradiance/AC-power correlation: {corr:.4f}. Highest mean-generation hour: {int(peak):02d}:00 in the dataset time standard. These are measured descriptive associations, not guarantees or causal findings.

```csv
{metrics_text}
```

Cleaning summary: {json.dumps(cleaning or {}, default=str)}

R² ≥0.95 and correlation >0.9 are hypotheses, not acceptance criteria. A short or synthetic dataset cannot establish annual or real-plant generalization. Temperature/normalized-power plots are observational proxies, not panel efficiency measurements; true conversion efficiency requires panel area and appropriate plane-of-array radiation.

## Conclusion
The pipeline supports reproducible, leakage-aware comparison. Practical suitability must be established with real plant data and future-weather inputs using an operational backtest.

## Future Scope
Real-time IoT integration; externally supplied weather forecasts for day-ahead prediction; battery dispatch optimization with tariffs and constraints; combined solar/wind forecasts; uncertainty intervals and rolling-origin evaluation. These are roadmap items, not implemented capabilities.

## References
- Kaggle dataset: {KAGGLE_URL}
- NASA POWER hourly API: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- MNRE capacity reports: https://mnre.gov.in/installed-capacity/
- CEA reports: https://cea.nic.in/reports/external/daily-generation?lang=en
- Scikit-learn regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- XGBoost: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- TensorFlow time-series tutorial: https://www.tensorflow.org/tutorials/structured_data/time_series
'''
