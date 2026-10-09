# ode-ablation-load-l1

## Summary: day-ahead forecast of net grid consumption

**Task.** Forecast the net electrical consumption from the grid 24 hours ahead, as
hourly mean power in watts, judged by the criteria in `evaluation.yaml` (RMSE at 1 h
resolution, threshold 6.1).

**Outcome.** The operator works end to end and is evaluated honestly, but **it does
not meet the threshold.** The best September RMSE is 219.5 W, about 36 times the
6.1 W threshold. Nothing on the platform, as far as this work found, closes that
gap, and whether 6.1 is the intended value (unit, target) is still an open question
for the owner of `evaluation.yaml`.

### How the data was found

- The ontology (Get-Power on the Grid / Consumption / Site aspects) resolved to no
  device: no grid meter on this platform carries that annotation. The only matching
  import type (SolarEdge) needs an API key and cannot be deployed from ODE.
- Two meters were then found in the device list: the household utility meter
  "Stromzähler" (Iskra MT 175, Tasmota) and a Qubino 3-phase meter. Profiles over
  June to August 2026 showed:
  - `sensor.MT175.P`: signed instantaneous power in W (negative values mean export),
    regular 10 s sampling, 0.99 completeness, a daily period, history since 2024-11.
  - Qubino total power: the same load events, but irregular and only 0.34 complete.
- **Selected series** (confirmed by the developer):
  `urn:infai:ses:device:8ae74f8f-07c9-42c2-ad1c-701b503adac1` /
  `urn:infai:ses:service:61447a7c-5e75-48dd-b2db-bcfcd23fa33d` / `sensor.MT175.P`.
  It is both the target and the operator's only input, which is what lets Operator
  Lib score a test window: the target has to be one of the operator's own inputs.
- Caveat: the profiles' raw reads were truncated to their row limit, so their
  distribution figures describe roughly the last 1.5 days before the split, not
  all 90 days.

### Model (`training.py`)

- Raw 10 s readings are aggregated in Ray to hourly means in UTC. An hour with fewer
  than 30 samples counts as missing.
- The forecast for target hour T is issued during hour H = T − 24 h and uses only
  hours that are complete by then, so training and inference see the same
  information and the horizon is a true 24 h:
  - `lag168`: same hour last week
  - `lag48`: same hour of day, two days earlier
  - `lag25`: last complete hour at forecast time
  - `level24`: mean of the 24 complete hours T−48 h … T−25 h
  - `profile`: mean by local hour of day (Europe/Berlin), weekday vs weekend, over
    the last `PROFILE_DAYS` days
- The coefficients are ordinary least squares over a 120-day window
  (`TRAINING_WINDOW_DAYS`). A missing lag falls back to the profile.
- The last 21 days (`VALIDATION_DAYS`) are held out first and the model is compared
  with naive baselines: `val_rmse`, `val_rmse_naive_lag168`, `val_rmse_naive_lag48`,
  `val_rmse_naive_lag25` and `val_rmse_profile`, all logged on the run.
- The model stores its coefficients, its profile and the last 8 days of hourly means
  at training end.

### Operator (`op.py`)

- Each message adds to its hour's running mean. History from before the first
  message comes from the model.
- On the first message of each new hour H, the operator forecasts T = H + 24 h and
  emits `{"prediction": W}` stamped with T, which is the time Operator Lib buckets
  on when it scores. That is one forecast per hour, not one per message.
- Retraining is time-based (`retrain_after_s`, default 1 day).

### Evaluation runs

All runs trained on history before 2026-09-01 and replayed 2026-09-01 to
2026-10-01: 243,762 messages, 696 forecasts, 648 hours scored. Of the 48 forecasts
not scored, 24 target hours after the test window. The other 24 most likely fall
on hours with no meter readings; this was not verified, because the test window was
not readable in the session.

| Run | Change | Model, validation | Profile only, validation | September RMSE |
|---|---|---|---|---|
| 1 (`5827fb5`) | baseline: lag168, lag48, level24, profile | 239.2 W | 240.7 W | 222.5 W |
| 2 (`c1179c5`) | + lag25 | 239.6 W | 240.7 W | 221.1 W |
| 3 (`c1179c5`, `PROFILE_DAYS=21`) | shorter profile window | 239.2 W | 238.2 W | 219.5 W |

What the runs show:

- The model is essentially the daily profile. The profile weight is 0.56 to 0.73,
  the lags carry under 0.1 each, and in run 3 the regression is slightly worse in
  validation than the profile alone.
- Every naive lag forecast is far worse (316 to 339 W in validation).
- Validation and September errors agree across all runs, so there is no sign of a
  mismatch between training and inference. The 1 to 3 W differences between runs
  are within noise.
- The remaining error is most likely PV output varying with the weather, plus
  large loads that switch on irregularly (peaks up to about 8.9 kW). The meter's own
  history cannot predict either.

### What was tried and ruled out

- **Day-ahead weather as a feature.**
  - The Open-Meteo forecast archive export holds no rows before the split. Its rows
    probably carry write-time timestamps instead of `issued_at`, which would need an
    export with `time_path`; the archive would also need to backfill again.
  - The yr.no forecast export reaches back only to about 2026-08-22: roughly 10
    days, against 120 days of meter history.
  - Neither can train a weather coefficient today.
- **Using the partially observed current hour.** Done literally, this would mean
  issuing the forecast later in the hour, which shortens the horizon below 24 h. It
  was replaced by `lag25`, which keeps the horizon honest.

### Open points

- **Threshold.** Confirm that `threshold: 6.1` is in the unit and on the target
  intended.
- **Weather.** For a real improvement, give the Open-Meteo archive a correctly
  timestamped export (`time_path` = `value.issued_at`), or let yr.no accumulate
  months of history. Then add forecast cloud cover or radiation for the target hour
  as a feature.
- **Simplification.** Since the regression adds nothing over the profile, a
  profile-only operator with `PROFILE_DAYS=21` would perform the same and be simpler.
- **Ontology.** Annotating the MT 175's `sensor.MT175.P` with the Grid aspect would
  make this meter findable through the ontology for the next person.

---

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.2". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-load-l1". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-load-l1:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.2", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
