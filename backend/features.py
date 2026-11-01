"""Inference feature contract copied from the verified Phase 1 implementation.

Keep this module aligned with temporal_features.py and data_cleaning.py when
retraining models. The backend image does not import training code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAGNETIC_COLUMNS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45",
    "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY",
    "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX", "R_VALUE",
]
BASE_COLUMNS = [f"{name}__last" for name in MAGNETIC_COLUMNS]
TEMPORAL_COLUMNS = [
    f"{name}__{stat}"
    for name in MAGNETIC_COLUMNS
    for stat in ("lag_1h", "lag_3h", "rate_3h", "mean_3h", "std_3h")
]


def compute_region_features(
    frame: pd.DataFrame, windows: list[dict[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return final-state and time-local values in saved feature order."""
    timestamps = pd.to_datetime(frame["Timestamp"], errors="raise").to_numpy(dtype="datetime64[ns]")
    if len(timestamps) == 0 or np.any(timestamps[1:] <= timestamps[:-1]):
        raise ValueError("Full-series timestamps must be strictly increasing")
    values = frame[MAGNETIC_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    starts = np.array([np.datetime64(row["start"]) for row in windows], dtype="datetime64[ns]")
    ends = np.array([np.datetime64(row["end"]) for row in windows], dtype="datetime64[ns]")
    start_idx = np.searchsorted(timestamps, starts)
    end_idx = np.searchsorted(timestamps, ends)
    valid = (
        (start_idx < len(timestamps))
        & (end_idx < len(timestamps))
        & (timestamps[np.minimum(start_idx, len(timestamps) - 1)] == starts)
        & (timestamps[np.minimum(end_idx, len(timestamps) - 1)] == ends)
        & (end_idx - start_idx + 1 == 60)
    )
    if not np.all(valid):
        example = windows[int(np.flatnonzero(~valid)[0])]["path"]
        raise ValueError(f"Instance does not match a 60-point full-series slice: {example}")

    latest = values[end_idx]
    base = pd.DataFrame(latest, columns=BASE_COLUMNS)
    time_index = pd.DatetimeIndex(timestamps)
    series = pd.DataFrame(values, index=time_index, columns=MAGNETIC_COLUMNS)
    rolling = series.rolling("3h", closed="both", min_periods=8)
    means = rolling.mean().to_numpy()[end_idx]
    stds = rolling.std(ddof=0).to_numpy()[end_idx]

    def lag(hours: int) -> np.ndarray:
        target = ends - np.timedelta64(hours, "h")
        idx = np.searchsorted(timestamps, target)
        valid_lag = (
            (idx < len(timestamps))
            & (timestamps[np.minimum(idx, len(timestamps) - 1)] == target)
            & (idx >= start_idx)
        )
        result = np.full_like(latest, np.nan)
        result[valid_lag] = values[idx[valid_lag]]
        return result

    lag_1h = lag(1)
    lag_3h = lag(3)
    rate_3h = (latest - lag_3h) / 3.0
    stacked = np.stack((lag_1h, lag_3h, rate_3h, means, stds), axis=2)
    temporal = pd.DataFrame(stacked.reshape(len(windows), -1), columns=TEMPORAL_COLUMNS)
    return base, temporal


def transform_features(values: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the saved training-only imputation and z-score normalization."""
    result = np.asarray(values, dtype=np.float64).copy()
    medians = np.asarray(stats["impute_median"], dtype=np.float64)
    means = np.asarray(stats["mean_after_imputation"], dtype=np.float64)
    scales = np.asarray(stats["scale_after_imputation"], dtype=np.float64)
    if result.shape[-1] != len(medians):
        raise ValueError("Feature count does not match the normalization file")
    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = np.broadcast_to(medians, result.shape)[invalid]
    return (result - means) / scales
