"""Time-local features for the shared 12-hour SWAN-SF instance windows.

Use `python data_cleaning.py` to build both baseline and temporal feature tables
in one pass over the full active-region series. This module owns the temporal
feature definitions and can also start that build directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Magnetic measurements only. Flare annotations, X-ray flux, and quality fields
# are excluded so future event information cannot enter these feature tables.
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
    """Return baseline and temporal values in the order of `windows`.

    The window ends and lags are matched by timestamp. A lag that is absent
    from the observation window becomes NaN and is imputed using training data.
    Rolling summaries use only the final three hours of each observation window.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    from data_cleaning import build_phase1

    build_phase1(args.data_dir)


if __name__ == "__main__":
    main()
