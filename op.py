"""The operator: day-ahead forecast of the hourly mean net grid power (W).

Every message adds to the running mean of its UTC hour. Once per hour (on the first
message of a new hour H) the operator forecasts target hour T = H + 24h from hours
that are complete by then, and emits {"prediction": W} stamped with T — the time
the forecast is about, which is what the evaluation buckets on.
"""

import datetime
import math
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import train_model, forecast_value, slot_of, LAGS


HOUR = datetime.timedelta(hours=1)


class CustomConfig(Config):
    # Retrain at most this often, in seconds.
    retrain_after_s = 86400


def _hour_floor(at: datetime.datetime) -> datetime.datetime:
    if at.tzinfo is None:
        at = at.replace(tzinfo=datetime.timezone.utc)
    at = at.astimezone(datetime.timezone.utc)
    return at.replace(minute=0, second=0, microsecond=0)


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._params: typing.Optional[dict] = None
        self._params_source: typing.Optional[object] = None
        self._history: typing.Dict[datetime.datetime, float] = {}
        self._sums: typing.Dict[datetime.datetime, float] = {}
        self._counts: typing.Dict[datetime.datetime, int] = {}
        self._last_target: typing.Optional[datetime.datetime] = None
        super().init(*args, **kwargs)

    # ------------------------------------------------------------------ state

    def _ensure_params(self, model: PyFuncModel) -> None:
        if self._params is not None and self._params_source is model:
            return
        params = model.predict({"op": "params"})
        self._params = params
        self._params_source = model
        self._history = {
            _hour_floor(datetime.datetime.fromisoformat(k)): float(v)
            for k, v in params.get("history", {}).items()
        }
        self._last_target = None

    def _hourly(self, bucket: datetime.datetime) -> typing.Optional[float]:
        n = self._counts.get(bucket, 0)
        if n >= self._params.get("min_samples_per_hour", 30):
            return self._sums[bucket] / n
        return self._history.get(bucket)

    def _forecast(self, target: datetime.datetime) -> float:
        profile = float(self._params["profile"][slot_of(target)])
        lags = self._params.get("lags", LAGS)
        features: typing.Dict[str, typing.Optional[float]] = {
            name: self._hourly(target - hours * HOUR) for name, hours in lags.items()
        }
        level = [self._hourly(target - k * HOUR) for k in range(25, 49)]
        level = [v for v in level if v is not None]
        features["level24"] = sum(level) / len(level) if len(level) >= 12 else None
        return forecast_value(self._params["coef"], features, profile)

    def _prune(self, now_bucket: datetime.datetime) -> None:
        horizon = now_bucket - 200 * HOUR
        for store in (self._sums, self._counts):
            for b in [b for b in store if b < horizon]:
                del store[b]
        for b in [b for b in self._history if b < horizon]:
            del self._history[b]

    # ------------------------------------------------------------------ operator API

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        if model is None:
            return None, None, None
        self._ensure_params(model)

        bucket = _hour_floor(timestamp)
        value = data.get("value")
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            self._sums[bucket] = self._sums.get(bucket, 0.0) + value
            self._counts[bucket] = self._counts.get(bucket, 0) + 1

        target = bucket + self._params.get("horizon_h", 24) * HOUR
        if target == self._last_target:
            return None, None, None
        self._last_target = target
        prediction = self._forecast(target)
        self._prune(bucket)
        return target, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
