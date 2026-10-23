"""Hourly solar analysis and weather-conditioned TensorFlow forecasting."""
import io
import zipfile
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER = ['irradiance_wm2', 'temperature_c', 'humidity_pct', 'cloud_cover_pct']


def demo_data(days=120, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2026-01-01', periods=days * 24, freq='h')
    sun = np.maximum(0, np.sin(np.pi * (dates.hour.to_numpy() - 6) / 12))
    cloud = np.clip(rng.normal(35, 24, len(dates)), 0, 100)
    radiation = sun * 950 * (1 - cloud / 130)
    temp = 24 + 8 * sun + rng.normal(0, 2, len(dates))
    power = np.maximum(0, radiation / 1000 * 100 * (1 - .004 * (temp - 25))
                       + rng.normal(0, 1.3, len(dates)))
    power[radiation == 0] = 0
    return pd.DataFrame(dict(timestamp=dates, irradiance_wm2=radiation,
                             temperature_c=temp, humidity_pct=np.clip(75 - 25 * sun + rng.normal(0, 8, len(dates)), 0, 100),
                             cloud_cover_pct=cloud, power_kw=power))


def validate(frame, history=True):
    required = ['timestamp'] + WEATHER + (['power_kw'] if history else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    df = frame[required].copy()
    df['timestamp'] = pd.to_datetime(df.timestamp, errors='coerce')
    if df.timestamp.isna().any():
        raise ValueError('Every timestamp must be a valid date and time.')
    if df.timestamp.dt.tz is not None:
        raise ValueError('Use timezone-free timestamps in the plant’s local time.')
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.timestamp.duplicated().any():
        raise ValueError('Duplicate timestamps found. Supply one plant reading per hour.')
    if len(df) < (240 if history else 1):
        raise ValueError('History needs at least 240 hourly rows.' if history else 'Supply at least one weather row.')
    if len(df) > 1 and not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError('Rows must be exactly one hour apart, without gaps.')
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} contains missing or non-finite values.')
    for col in ['irradiance_wm2'] + (['power_kw'] if history else []):
        if (df[col] < 0).any():
            raise ValueError(f'{col} cannot be negative.')
    for col in ['humidity_pct', 'cloud_cover_pct']:
        if not df[col].between(0, 100).all():
            raise ValueError(f'{col} must be between 0 and 100.')
    return df


def features(df):
    hour = df.timestamp.dt.hour.to_numpy()
    day = df.timestamp.dt.dayofyear.to_numpy()
    return np.column_stack([df[WEATHER].to_numpy(), np.sin(hour * 2 * np.pi / 24),
                            np.cos(hour * 2 * np.pi / 24), np.sin(day * 2 * np.pi / 365.25),
                            np.cos(day * 2 * np.pi / 365.25)]).astype('float32')


def metrics(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    return {'MAE (kW)': float(np.mean(np.abs(error))), 'RMSE (kW)': float(np.sqrt(np.mean(error ** 2)))}


def predict(bundle, weather):
    x = (features(weather) - bundle['mean']) / bundle['std']
    result = np.maximum(0, np.asarray(bundle['model'](x, training=False)).ravel() * bundle['target_scale'])
    result[weather.irradiance_wm2.to_numpy() <= 0] = 0
    return result


def train(df, epochs=35):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    x = features(df)
    y = df.power_kw.to_numpy(dtype='float32')
    a, b = int(len(df) * .7), int(len(df) * .85)
    mean, std = x[:a].mean(axis=0), x[:a].std(axis=0)
    std[std < 1e-6] = 1
    scale = max(float(y[:a].max()), 1.)
    z = (x - mean) / std
    model = tf.keras.Sequential([tf.keras.layers.Input((x.shape[1],)),
                                 tf.keras.layers.Dense(64, activation='relu'),
                                 tf.keras.layers.Dense(32, activation='relu'),
                                 tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='mse')
    history = model.fit(z[:a], y[:a] / scale, validation_data=(z[a:b], y[a:b] / scale),
                        epochs=epochs, batch_size=64, shuffle=False, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)])
    bundle = dict(model=model, mean=mean, std=std, target_scale=scale)
    test = df.iloc[b:].copy()
    test['predicted_kw'] = predict(bundle, test)
    test['previous_day_kw'] = df.power_kw.shift(24).iloc[b:].to_numpy()
    return bundle, test, pd.DataFrame(history.history), {'train': a, 'validation': b-a, 'test': len(df)-b}


def weather_scenario(df, hours):
    """Prior-seven-day hourly weather averages; explicitly a scenario, not weather prediction."""
    dates = pd.date_range(df.timestamp.iloc[-1] + pd.Timedelta(hours=1), periods=hours, freq='h')
    profile = df.tail(168).groupby(df.tail(168).timestamp.dt.hour)[WEATHER].mean()
    result = profile.loc[dates.hour].reset_index(drop=True)
    result.insert(0, 'timestamp', dates)
    return result


def export_model(bundle):
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        path = Path(directory) / 'model.keras'
        bundle['model'].save(path)
        metadata = dict(mean=bundle['mean'].tolist(), std=bundle['std'].tolist(),
                        target_scale=bundle['target_scale'], weather_columns=WEATHER,
                        feature_order=WEATHER + ['hour_sin', 'hour_cos', 'year_sin', 'year_cos'])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(path, 'model.keras')
            archive.writestr('preprocessing.json', json.dumps(metadata, indent=2))
        return buffer.getvalue()
