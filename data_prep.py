"""Turn NASA POWER daily T2M_MAX into Poisson (weekly counts) and
exponential (gaps between events) data."""
import pandas as pd


def load_daily(path_or_buffer):
    df = pd.read_csv(path_or_buffer, parse_dates=["date"])
    df = df[df["tmax"] > -900]          # NASA POWER uses -999 for missing
    return df.sort_values("date").reset_index(drop=True)


def mark_events(df, threshold=40.0, months=(4, 5, 6)):
    """Keep the chosen season and flag event days (tmax > threshold)."""
    d = df[df["date"].dt.month.isin(months)].copy()
    d["year"] = d["date"].dt.year
    d["event"] = d["tmax"] > threshold
    return d


def weekly_counts(d, months=(4, 5, 6)):
    """Events per full 7-day week, counted from the 1st day of the first month."""
    d = d.copy()
    start = pd.to_datetime(d["year"].astype(str) + f"-{months[0]:02d}-01")
    d["week"] = (d["date"] - start).dt.days // 7
    g = d.groupby(["year", "week"]).agg(days=("event", "size"),
                                        events=("event", "sum")).reset_index()
    g = g[g["days"] == 7].drop(columns="days")   # drop incomplete weeks
    g["events"] = g["events"].astype(int)
    return g


def event_gaps(d):
    """Days between consecutive event days, computed within each year."""
    gaps = []
    for _, grp in d[d["event"]].groupby("year"):
        gaps.extend(grp["date"].diff().dt.days.dropna().astype(int).tolist())
    return gaps


# ---------------------------------------------------------------------------
# Battery discharge-cycle data (columns: cycle, time, temperature, thermal_event)
# thermal_event = 1 for every reading with temperature > 40 deg C.
# ---------------------------------------------------------------------------
def is_battery_file(df):
    return {"cycle", "time", "temperature", "thermal_event"}.issubset(df.columns)


def load_battery(path_or_buffer):
    return pd.read_csv(path_or_buffer).sort_values(["cycle", "time"]).reset_index(drop=True)


def battery_cycle_table(df, threshold=None):
    """One row per discharge cycle: number of event readings, peak temperature,
    duration, time of first event reading (NaN if none).
    If threshold is given, events are recomputed as temperature > threshold;
    otherwise the file's own thermal_event column is used."""
    d = df.copy()
    if threshold is not None:
        d["thermal_event"] = (d["temperature"] > threshold).astype(int)
    g = d.groupby("cycle")
    out = pd.DataFrame({
        "events": g["thermal_event"].sum().astype(int),
        "readings": g.size(),
        "peak_temp": g["temperature"].max(),
        "duration_s": g["time"].max(),
    })
    out["first_event_s"] = d[d["thermal_event"] == 1].groupby("cycle")["time"].min()
    return out.reset_index()


def battery_stage_table(cycles, edges=(0, 150, 300, 400, 10_000)):
    """Mean / variance / dispersion of event counts per stage of battery life."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = cycles[(cycles["cycle"] > lo) & (cycles["cycle"] <= hi)]["events"]
        if len(x) > 1:
            m, v = x.mean(), x.var(ddof=1)
            rows.append({"cycles": f"{int(x.index.size)} cycles ({lo + 1}-{min(hi, int(cycles['cycle'].max()))})",
                         "mean": m, "variance": v,
                         "dispersion": v / m if m > 0 else float("nan")})
    return pd.DataFrame(rows)
