"""
Statistical & Data Processing Core for Battery Thermal Event Simulator
======================================================================
Provides validated telemetry loading, episode onset extraction,
Garwood exact Poisson confidence intervals, cycle-level aggregations,
and reproducible Monte Carlo simulation routines.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class DataValidationError(Exception):
    """Raised when an input dataset fails structural or statistical validation."""
    pass


def validate_and_load_dataframe(
    source: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, int, Dict[str, Any]]:
    """Validate and process a battery thermal telemetry CSV dataset.

    Parameters
    ----------
    source : str, Path, or file-like object
        Path to CSV file or uploaded BytesIO buffer.

    Returns
    -------
    df : pd.DataFrame
        Cleaned telemetry DataFrame with added `global_s` column.
    onsets : pd.DataFrame
        Telemetry rows corresponding to episode onsets.
    total_s : float
        Total operational exposure in seconds across all cycles.
    n_cycles : int
        Number of unique battery cycles analyzed.
    report : dict
        Validation summary report.
    """
    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_csv(source)
    except Exception as e:
        raise DataValidationError(f"Unable to parse CSV file: {str(e)}")

    if df.empty:
        raise DataValidationError("The supplied CSV file is empty (0 records).")

    # Standardize column headers (strip whitespace and convert to lowercase)
    orig_cols = list(df.columns)
    df.rename(columns={col: col.strip().lower() for col in df.columns}, inplace=True)

    required_cols = {"cycle", "time", "temperature", "thermal_event"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {sorted(list(missing))}. "
            f"Expected schema: ['cycle', 'time', 'temperature', 'thermal_event']. "
            f"Found: {orig_cols}"
        )

    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "warnings": [],
        "deduplicated_rows": 0,
        "null_rows_dropped": 0,
    }

    # 1. Missing / NaN check
    null_count = int(df[list(required_cols)].isnull().sum().sum())
    if null_count > 0:
        report["warnings"].append(f"Dropped {null_count} rows with null/missing values.")
        df.dropna(subset=list(required_cols), inplace=True)
        report["null_rows_dropped"] = null_count
        if df.empty:
            raise DataValidationError("Dataset contains only null entries; no valid records remain.")

    # 2. Type coercion & verification
    try:
        df["cycle"] = pd.to_numeric(df["cycle"], errors="raise").astype(int)
    except Exception:
        raise DataValidationError("Column 'cycle' must contain integer cycle numbers.")

    try:
        df["time"] = pd.to_numeric(df["time"], errors="raise").astype(float)
    except Exception:
        raise DataValidationError("Column 'time' must contain numeric timestamps in seconds.")

    try:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="raise").astype(float)
    except Exception:
        raise DataValidationError("Column 'temperature' must contain numeric temperature values.")

    try:
        df["thermal_event"] = pd.to_numeric(df["thermal_event"], errors="raise").astype(int)
    except Exception:
        raise DataValidationError("Column 'thermal_event' must contain binary flags (0 or 1).")

    # 3. Value boundary checks
    if (df["time"] < 0).any():
        report["warnings"].append("Found negative time values. Clocks may be uncalibrated.")

    invalid_flags = ~df["thermal_event"].isin([0, 1])
    if invalid_flags.any():
        bad_vals = df.loc[invalid_flags, "thermal_event"].unique()
        raise DataValidationError(
            f"Invalid values in 'thermal_event': {bad_vals}. Only 0 and 1 are permitted."
        )

    # 4. Duplicate checks
    dups = int(df.duplicated(subset=["cycle", "time"]).sum())
    if dups > 0:
        report["warnings"].append(f"Removed {dups} duplicate records with identical (cycle, time).")
        df.drop_duplicates(subset=["cycle", "time"], keep="first", inplace=True)
        report["deduplicated_rows"] = dups

    # 5. Sorting and Time Ordering
    df.sort_values(by=["cycle", "time"], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6. Cycle length & global timeline stitching
    cycle_len = df.groupby("cycle")["time"].max()
    cycle_start = cycle_len.cumsum() - cycle_len
    df["global_s"] = df["time"] + df["cycle"].map(cycle_start)

    # 7. Episode Onset Extraction
    # An episode is a consecutive run of thermal_event == 1 within the same cycle.
    # An onset is the initial 0 -> 1 transition.
    df["prev_flag"] = df.groupby("cycle")["thermal_event"].shift(fill_value=0)
    onsets = df[(df["thermal_event"] == 1) & (df["prev_flag"] == 0)].copy()

    total_s = float(cycle_len.sum())
    n_cycles = int(cycle_len.size)

    report["final_rows"] = len(df)
    report["cycles_count"] = n_cycles
    report["flagged_readings"] = int(df["thermal_event"].sum())
    report["episode_onsets"] = int(len(onsets))

    return df, onsets, total_s, n_cycles, report


def extract_episodes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Extract detailed episode records including onset time, duration, and peak temperature."""
    df_work = df.copy()
    df_work["prev_flag"] = df_work.groupby("cycle")["thermal_event"].shift(fill_value=0)
    df_work["is_onset"] = (df_work["thermal_event"] == 1) & (df_work["prev_flag"] == 0)
    df_work["episode_id"] = df_work["is_onset"].cumsum()
    df_work.loc[df_work["thermal_event"] == 0, "episode_id"] = 0

    flagged_df = df_work[df_work["thermal_event"] == 1]
    if flagged_df.empty:
        return pd.DataFrame(columns=[
            "episode_id", "cycle", "onset_time_s", "onset_global_h",
            "duration_readings", "duration_s", "peak_temp_c", "wait_from_prev_h"
        ])

    episodes = []
    for ep_id, grp in flagged_df.groupby("episode_id"):
        cycle = int(grp["cycle"].iloc[0])
        onset_time_s = float(grp["time"].iloc[0])
        global_s = float(grp["global_s"].iloc[0])
        duration_readings = int(len(grp))
        duration_s = float(grp["time"].iloc[-1] - grp["time"].iloc[0])
        peak_temp = float(grp["temperature"].max())
        episodes.append({
            "episode_id": int(ep_id),
            "cycle": cycle,
            "onset_time_s": onset_time_s,
            "onset_global_h": global_s / 3600.0,
            "duration_readings": duration_readings,
            "duration_s": duration_s,
            "peak_temp_c": peak_temp,
        })

    ep_df = pd.DataFrame(episodes)
    ep_df["wait_from_prev_h"] = ep_df["onset_global_h"].diff()
    return ep_df


def garwood_poisson_ci(n: int, hours: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Calculate exact Garwood Poisson arrival rate confidence interval."""
    if hours <= 0:
        return (0.0, float("inf"))
    lam_lo = float(stats.chi2.ppf(alpha / 2.0, 2 * n) / (2.0 * hours)) if n > 0 else 0.0
    lam_hi = float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1)) / (2.0 * hours))
    return lam_lo, lam_hi


def data_summary(
    df: pd.DataFrame,
    onsets: pd.DataFrame,
    total_s: float,
    n_cycles: int,
) -> Dict[str, Any]:
    """Calculate summary statistics, arrival rate lambda, and confidence intervals."""
    hours = total_s / 3600.0 if total_s > 0 else 1e-9
    n_episodes = len(onsets)
    n_flagged = int(df["thermal_event"].sum())
    pct_flagged = (n_flagged / len(df) * 100.0) if len(df) > 0 else 0.0

    lam = n_episodes / hours
    lam_lo, lam_hi = garwood_poisson_ci(n_episodes, hours)

    # Per-cycle event count dispersion
    unique_cycles = df["cycle"].unique()
    per_cycle = onsets.groupby("cycle").size().reindex(unique_cycles, fill_value=0)
    mean_c = float(per_cycle.mean())
    var_c = float(per_cycle.var(ddof=1)) if len(per_cycle) > 1 else 0.0
    dispersion = var_c / mean_c if mean_c > 0 else float("nan")

    # Overall peak temperature
    peak_temp = float(df["temperature"].max())

    return {
        "total_rows": len(df),
        "flagged_rows": n_flagged,
        "pct_flagged": pct_flagged,
        "episodes": n_episodes,
        "cycles": n_cycles,
        "hours": hours,
        "total_s": total_s,
        "lam": lam,
        "lam_lo": lam_lo,
        "lam_hi": lam_hi,
        "per_cycle_mean": mean_c,
        "per_cycle_var": var_c,
        "per_cycle_dispersion": dispersion,
        "per_cycle_counts": per_cycle,
        "peak_temperature": peak_temp,
    }


def waiting_times_hours(onsets: pd.DataFrame) -> np.ndarray:
    """Calculate inter-arrival gaps (hours) between consecutive episode onsets."""
    if len(onsets) < 2:
        return np.array([])
    t_h = np.sort(onsets["global_s"].to_numpy()) / 3600.0
    return np.diff(t_h)


def cycle_diagnostics(df: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cycle telemetry statistics, duration, and episode counts."""
    cycle_stats = df.groupby("cycle").agg(
        readings=("temperature", "count"),
        duration_s=("time", "max"),
        min_temp=("temperature", "min"),
        mean_temp=("temperature", "mean"),
        max_temp=("temperature", "max"),
        flagged_readings=("thermal_event", "sum"),
    ).reset_index()

    cycle_stats["duration_h"] = cycle_stats["duration_s"] / 3600.0

    onset_counts = onsets.groupby("cycle").size().reset_index(name="episodes")
    cycle_stats = pd.merge(cycle_stats, onset_counts, on="cycle", how="left")
    cycle_stats["episodes"] = cycle_stats["episodes"].fillna(0).astype(int)
    cycle_stats["has_episode"] = (cycle_stats["episodes"] > 0).astype(int)

    return cycle_stats


def poisson_window(lam: float, t: float, k: int = 1) -> Dict[str, float]:
    """Calculate theoretical Poisson probabilities for count N in window of length t (hours)."""
    mu = float(lam * t)
    return {
        "mu": mu,
        "p0": float(stats.poisson.pmf(0, mu)),
        "p_ge_1": float(stats.poisson.sf(0, mu)),
        "p_eq_k": float(stats.poisson.pmf(k, mu)),
        "p_ge_k": float(stats.poisson.sf(k - 1, mu)),
    }


def exponential_wait(lam: float, t: float) -> Dict[str, float]:
    """Calculate theoretical Exponential waiting time quantities for rate lam."""
    mean_wait = 1.0 / lam if lam > 0 else float("inf")
    return {
        "mean": mean_wait,
        "variance": 1.0 / (lam**2) if lam > 0 else float("inf"),
        "std": mean_wait,
        "median": float(np.log(2.0) / lam) if lam > 0 else float("inf"),
        "p_le_t": float(-np.expm1(-lam * t)) if lam > 0 else 0.0,
        "p_gt_t": float(np.exp(-lam * t)) if lam > 0 else 1.0,
    }


def evaluate_risk(p_risk: float, warn_threshold: float, high_threshold: float) -> str:
    """Classify risk level into LOW, MODERATE, or HIGH."""
    if p_risk >= high_threshold:
        return "HIGH RISK"
    if p_risk >= warn_threshold:
        return "MODERATE RISK"
    return "LOW RISK"


def simulate_counts(lam: float, t: float, n_windows: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n_windows Monte Carlo Poisson counts for window length t."""
    return rng.poisson(lam * t, size=n_windows)


def simulate_waits(lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n Exponential inter-arrival waiting times with rate lam."""
    return rng.exponential(1.0 / lam, size=n)


def simulate_paths(
    lam: float,
    duration: float,
    n_paths: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Generate discrete event arrival timestamps for n_paths sample battery timelines."""
    paths = []
    for _ in range(n_paths):
        arrivals = []
        cur_t = 0.0
        while True:
            dt = rng.exponential(1.0 / lam)
            cur_t += dt
            if cur_t > duration:
                break
            arrivals.append(cur_t)
        paths.append(np.array(arrivals))
    return paths