"""Training, on Ray.

Day-ahead forecast of the hourly mean of net grid power (W).

The forecast for target hour T is made during hour H = T - 24h, and it uses only
hours that are complete by then (<= H - 1), so training and inference see the
same information:

  lag168   hourly mean at T - 168h (same hour last week)
  lag48    hourly mean at T - 48h  (same hour of day, two days earlier)
  level24  mean of the 24 complete hours T-48h .. T-25h (recent level)
  profile  mean by local hour of day, weekday vs weekend

A missing lag falls back to the profile. The coefficients are ordinary least
squares. Before the final fit, the last VALIDATION_DAYS of the window are held out
and the model is compared with three naive baselines; those numbers are logged on
the run so a change can be judged against them.

op.py imports the helpers below so that both sides build features identically.
"""

import datetime
import os
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


TRAINING_WINDOW = datetime.timedelta(days=int(os.environ.get("TRAINING_WINDOW_DAYS", "120")))
VALIDATION_DAYS = int(os.environ.get("VALIDATION_DAYS", "21"))
# The profile follows the season better from recent weeks than from the whole window.
PROFILE_DAYS = int(os.environ.get("PROFILE_DAYS", "42"))
HORIZON_H = 24
LOCAL_TZ = "Europe/Berlin"
# At the meter's 10 s cadence an hour holds about 360 samples; an hour with fewer
# than this is treated as missing rather than as a mean of a few readings.
MIN_SAMPLES_PER_HOUR = 30
HISTORY_HOURS = 8 * 24
FEATURES = ("lag168", "lag48", "level24", "profile")


# --------------------------------------------------------------------------- shared helpers

def slot_of_index(index: pd.DatetimeIndex) -> np.ndarray:
    """0..47 per timestamp: weekday hours 0-23, weekend hours 24-47, local time."""
    local = index.tz_convert(LOCAL_TZ)
    weekend = (local.dayofweek >= 5).astype(int)
    return np.asarray(weekend * 24 + local.hour, dtype=int)


def slot_of(at: datetime.datetime) -> int:
    return int(slot_of_index(pd.DatetimeIndex([pd.Timestamp(at)]))[0])


def forecast_value(coef: typing.Sequence[float], lag168, lag48, level24, profile: float) -> float:
    """The one formula both training and inference use."""
    lag168 = profile if lag168 is None else lag168
    lag48 = profile if lag48 is None else lag48
    level24 = profile if level24 is None else level24
    return float(coef[0] + coef[1] * lag168 + coef[2] * lag48 + coef[3] * level24 + coef[4] * profile)


# --------------------------------------------------------------------------- fitting

def _hourly_partial(batch: pd.DataFrame) -> pd.DataFrame:
    """Per-batch hourly sum and count; combined in the driver."""
    if "value" not in batch or len(batch) == 0:
        return pd.DataFrame({"hour": pd.Series([], dtype="datetime64[ns]"),
                             "s": pd.Series([], dtype=float), "n": pd.Series([], dtype=int)})
    hour = pd.to_datetime(batch["time"], utc=True).dt.floor("1h").dt.tz_localize(None)
    value = pd.to_numeric(batch["value"].astype(object), errors="coerce").astype(float)
    frame = pd.DataFrame({"hour": hour, "v": value}).dropna()
    grouped = frame.groupby("hour")["v"].agg(["sum", "count"]).reset_index()
    return grouped.rename(columns={"sum": "s", "count": "n"})


def hourly_series(partials: pd.DataFrame) -> pd.Series:
    """Hourly mean in UTC, NaN where an hour has too few samples."""
    if partials.empty:
        return pd.Series(dtype=float)
    agg = partials.groupby("hour")[["s", "n"]].sum()
    idx = pd.DatetimeIndex(agg.index).tz_localize("UTC")
    mean = pd.Series(agg["s"].values / agg["n"].values, index=idx)
    mean[agg["n"].values < MIN_SAMPLES_PER_HOUR] = np.nan
    full = pd.date_range(idx.min(), idx.max(), freq="1h", tz="UTC")
    return mean.reindex(full)


def fit_profile(y: pd.Series) -> np.ndarray:
    y = y.dropna()
    overall = float(y.mean()) if len(y) else 0.0
    profile = np.full(48, overall)
    if not len(y):
        return profile
    slots = slot_of_index(y.index)
    values = y.values
    for k in range(48):
        sel = slots == k
        if sel.sum() >= 3:
            profile[k] = float(values[sel].mean())
    return profile


def feature_frame(y: pd.Series, profile: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame(index=y.index)
    f["profile"] = profile[slot_of_index(y.index)]
    f["lag168"] = y.shift(168).fillna(f["profile"])
    f["lag48"] = y.shift(48).fillna(f["profile"])
    f["level24"] = y.rolling(24, min_periods=12).mean().shift(25).fillna(f["profile"])
    f["y"] = y
    return f


def fit_coef(f: pd.DataFrame) -> np.ndarray:
    d = f.dropna(subset=["y"])
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in FEATURES])
    coef, *_ = np.linalg.lstsq(X, d["y"].values, rcond=None)
    return coef


def predict_frame(coef: np.ndarray, f: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(f))] + [f[c].values for c in FEATURES])
    return X @ coef


def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    ok = ~np.isnan(actual) & ~np.isnan(pred)
    return float(np.sqrt(np.mean((pred[ok] - actual[ok]) ** 2))) if ok.any() else float("nan")


def fit(y: pd.Series) -> typing.Tuple[dict, dict]:
    """Validate on the tail, then refit on everything. Returns (model params, metrics)."""
    metrics: typing.Dict[str, float] = {}
    end = y.index.max() + pd.Timedelta(hours=1)
    split = end - pd.Timedelta(days=VALIDATION_DAYS)

    fit_part = y[y.index < split]
    if fit_part.notna().sum() > 14 * 24:
        profile_v = fit_profile(fit_part[fit_part.index >= split - pd.Timedelta(days=PROFILE_DAYS)])
        f_all = feature_frame(y, profile_v)
        coef_v = fit_coef(f_all[f_all.index < split])
        val = f_all[f_all.index >= split]
        actual = val["y"].values
        metrics["val_rmse"] = _rmse(predict_frame(coef_v, val), actual)
        metrics["val_rmse_naive_lag168"] = _rmse(val["lag168"].values, actual)
        metrics["val_rmse_naive_lag48"] = _rmse(val["lag48"].values, actual)
        metrics["val_rmse_profile"] = _rmse(val["profile"].values, actual)
        metrics["val_hours"] = float(np.sum(~np.isnan(actual)))

    profile = fit_profile(y[y.index >= end - pd.Timedelta(days=PROFILE_DAYS)])
    f = feature_frame(y, profile)
    coef = fit_coef(f)
    metrics["train_rmse"] = _rmse(predict_frame(coef, f), f["y"].values)
    metrics["train_hours"] = float(y.notna().sum())
    for name, value in zip(("intercept",) + FEATURES, coef):
        metrics[f"coef_{name}"] = float(value)

    tail = y[y.index >= end - pd.Timedelta(hours=HISTORY_HOURS)].dropna()
    params = {
        "coef": [float(c) for c in coef],
        "profile": [float(p) for p in profile],
        "history": {ts.isoformat(): float(v) for ts, v in tail.items()},
        "horizon_h": HORIZON_H,
        "min_samples_per_hour": MIN_SAMPLES_PER_HOUR,
        "local_tz": LOCAL_TZ,
    }
    return params, metrics


# --------------------------------------------------------------------------- model

class OdeAblationLoadL1Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict({"op": "params"}) returns the fitted parameters and the hourly history
    at training end; op.py reads them once per model and forecasts itself, so the
    per-message path never goes through pyfunc.
    """

    def __init__(self, params: dict) -> None:
        self.params = params

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        if isinstance(payload, dict) and payload.get("op") == "forecast":
            return forecast_value(self.params["coef"], payload.get("lag168"), payload.get("lag48"),
                                  payload.get("level24"), float(payload["profile"]))
        return self.params


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        refs = provide_historic_data(TRAINING_WINDOW)
    if not refs:
        return None

    with logger.trace("hourly aggregation"):
        partials = []
        for ref in refs:
            ds = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
            partials.append(ds.map_batches(_hourly_partial, batch_format="pandas").to_pandas())
        partials = [p for p in partials if not p.empty]
        if not partials:
            return None
        y = hourly_series(pd.concat(partials, ignore_index=True))
    if y.notna().sum() < 8 * 24:
        # Too little to fit lags of a week; keep whatever model is registered.
        return None

    with logger.trace("fit"):
        params, metrics = fit(y)

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_days": VALIDATION_DAYS,
        "profile_days": PROFILE_DAYS,
        "horizon_h": HORIZON_H,
        "features": ",".join(FEATURES),
        "local_tz": LOCAL_TZ,
    })
    logger.log_metrics({k: v for k, v in metrics.items() if v == v})
    return OdeAblationLoadL1Model(params)
