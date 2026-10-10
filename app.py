"""
Interactive demo - Group 11 - Topic 11 (Poisson & Exponential battery events)
Flow shown on screen:  Input/Data -> Statistical Method -> Result -> AI Decision
Run:  streamlit run app.py

Works with:
  * battery discharge-cycle files (cycle, time, temperature, thermal_event)  <- main dataset
  * daily temperature files (date, tmax)   (NASA POWER / simulated)
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import data_prep as dp
import stats_core as sc

st.set_page_config(page_title="Battery Thermal Events - Poisson & Exponential",
                   layout="wide")
st.title("EV Battery Thermal Risk: Poisson and Exponential Models")
st.caption("IA-1 STAT-AI Engineering Challenge | Group 11 | "
           "Input/Data -> Statistical Method -> Result -> AI/Engineering Decision")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Input / Data")
files = sorted(glob.glob("*.csv"))
default = next((i for i, f in enumerate(files) if "B0005" in f), 0)
choice = st.sidebar.selectbox("Dataset", files + ["Upload my own CSV"], index=default)
if choice == "Upload my own CSV":
    up = st.sidebar.file_uploader("CSV", type="csv")
    if up is None:
        st.info("Upload a battery CSV (cycle, time, temperature, thermal_event) "
                "or a daily CSV (date, tmax).")
        st.stop()
    raw = pd.read_csv(up)
else:
    raw = pd.read_csv(choice)
is_sim = "SIMULATED" in str(choice).upper()
if is_sim:
    st.warning("SIMULATED demo data - not real measurements.")

battery_mode = dp.is_battery_file(raw)

# ---------------------------------------------------------- build the data
if battery_mode:
    st.sidebar.markdown("**Event = reading above threshold**")
    thr = st.sidebar.slider("Threshold (deg C)", 38.0, 41.0, 40.0, 0.5)
    cyc_all = dp.battery_cycle_table(raw, thr)
    lo, hi = int(cyc_all.cycle.min()), int(cyc_all.cycle.max())
    win = st.sidebar.slider("Cycle window analysed", lo, hi, (400, hi), step=2)
    cyc = cyc_all[(cyc_all.cycle >= win[0]) & (cyc_all.cycle <= win[1])]
    counts = cyc["events"].tolist()
    gaps = cyc["first_event_s"].dropna().tolist()
    count_label, wait_label, wait_unit = "Event readings per discharge cycle", \
        "Time from start of discharge to first event reading", "s"
    if win != (lo, hi):
        st.info(f"Window {win[0]}-{win[1]} chosen because the event rate changes with "
                "battery age (see the stage table in Step 2).")
else:
    daily = dp.load_daily(__import__("io").StringIO(raw.to_csv(index=False)))
    thr = st.sidebar.slider("Abnormal-event threshold (deg C)", 35.0, 48.0, 40.0, 0.5)
    months = st.sidebar.multiselect("Months included", list(range(1, 13)), default=[4, 5, 6])
    if not months:
        st.stop()
    months = tuple(sorted(months))
    d = dp.mark_events(daily, thr, months)
    wc = dp.weekly_counts(d, months)
    counts = wc["events"].tolist()
    gaps = dp.event_gaps(d)
    count_label, wait_label, wait_unit = "Event days per week", "Days between event days", "days"

st.sidebar.header("4. Decision rule")
k_default = 10 if battery_mode else 3
k_thr = st.sidebar.number_input("Alert if events per interval >= k", 1, 60, k_default)
p_thr = st.sidebar.slider("... with probability above", 0.05, 0.95, 0.50 if battery_mode else 0.25, 0.05)

if len(counts) < 20:
    st.error(f"Only {len(counts)} observations; the IA needs at least 20. Widen the window.")
    st.stop()
lam = sc.estimate_lambda(counts)
var = sc.sample_variance(counts)
if lam == 0:
    st.error("No events in this selection - lower the threshold or widen the window.")
    st.stop()

# -------------------------------------------------------------------- step 1
st.header("Step 1 - Input data")
c1, c2, c3 = st.columns(3)
if battery_mode:
    c1.metric("Temperature readings", f"{len(raw):,}")
    c2.metric("Event readings in window", int(sum(counts)))
    c3.metric("Discharge cycles (n)", len(counts))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(cyc_all["cycle"], cyc_all["events"], width=1.6, color="lightgray", label="outside window")
    ax.bar(cyc["cycle"], cyc["events"], width=1.6, color="tab:red", label="analysed window")
    ax.set_xlabel("Discharge cycle number"); ax.set_ylabel("Event readings per cycle")
    ax.legend(loc="upper left"); st.pyplot(fig)
    with st.expander("Show per-cycle table"):
        st.dataframe(cyc)
else:
    c1.metric("Daily records used", len(d))
    c2.metric("Event days (T > threshold)", int(d["event"].sum()))
    c3.metric("Weekly observations (n)", len(counts))
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(d["date"], d["tmax"], ".", ms=2, color="tab:gray")
    ax.scatter(d.loc[d["event"], "date"], d.loc[d["event"], "tmax"], s=4, color="tab:red")
    ax.axhline(thr, color="k", ls="--", lw=1)
    ax.set_ylabel("Daily max temp (C)"); st.pyplot(fig)
    with st.expander("Show weekly count table"):
        st.dataframe(wc)

# -------------------------------------------------------------------- step 2
st.header("Step 2 - Statistical method")
st.markdown(r"""
**Poisson** (counts per interval): $P(X=k)=\dfrac{e^{-\lambda}\lambda^k}{k!}$, with $E[X]=Var(X)=\lambda$, and $\hat\lambda=\bar x$  
**Exponential** (waiting time): $P(T>t)=e^{-\lambda t}$, $E[T]=1/\lambda$, and $CV=\sigma/\mu = 1$
""")
m1, m2, m3 = st.columns(3)
m1.metric("lambda = sample mean", f"{lam:.4f}")
m2.metric("Sample variance s^2", f"{var:.4f}")
m3.metric("Dispersion s^2 / mean", f"{var / lam:.3f}",
          help="About 1 supports Poisson. Above 1: clustering/trend. Below 1: events more regular than random.")
if var / lam > 1.3:
    st.warning("Variance is much larger than the mean (overdispersion): the rate is not constant "
               "or events cluster, so a single Poisson model fits poorly here.")
elif var / lam < 0.7:
    st.warning("Variance is smaller than the mean (underdispersion): counts are more regular "
               "than a Poisson process would give.")
if battery_mode:
    st.subheader("Event rate by stage of battery life")
    st.dataframe(dp.battery_stage_table(cyc_all).style.format(
        {"mean": "{:.2f}", "variance": "{:.2f}", "dispersion": "{:.2f}"}))
    st.caption("The rate rises sharply with battery age, so one lambda over all cycles is not valid.")

# -------------------------------------------------------------------- step 3
st.header("Step 3 - Results")
left, right = st.columns(2)
with left:
    st.subheader("Poisson fit")
    ks = np.arange(0, max(counts) + 3)
    obs = [counts.count(int(i)) / len(counts) for i in ks]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(ks - 0.2, obs, 0.4, label="Observed")
    ax.bar(ks + 0.2, [sc.poisson_pmf(int(i), lam) for i in ks], 0.4, label="Poisson")
    ax.set_xlabel(count_label); ax.set_ylabel("Relative frequency"); ax.legend()
    st.pyplot(fig)
    rows = sc.observed_vs_expected(counts, lam)
    stat, dof, p = sc.chi_square_gof(rows)
    tbl = pd.DataFrame(rows).rename(columns={"label": "k", "chi_term": "(O-E)^2/E"})
    st.dataframe(tbl.style.format({"expected": "{:.2f}", "(O-E)^2/E": "{:.3f}"}))
    if p is not None:
        st.write(f"Chi-square = **{stat:.2f}**, dof = {dof}, p-value = **{p:.4f}** "
                 f"({'fit rejected at 5%' if p < 0.05 else 'fit not rejected at 5%'})")
    st.markdown("**Probability calculator**")
    k = st.number_input("k", 0, 60, int(round(lam)), key="k_calc")
    st.write(f"P(X = {k}) = **{sc.poisson_pmf(int(k), lam):.4f}**   |   "
             f"P(X >= {k}) = **{sc.poisson_sf_ge(int(k), lam):.4f}**")

with right:
    st.subheader("Exponential waiting time")
    if len(gaps) >= 5:
        mu = float(np.mean(gaps)); sd = float(np.std(gaps, ddof=1))
        lam_t = 1 / mu
        st.write(f"{wait_label}: mean = **{mu:.1f} {wait_unit}** -> lambda = {lam_t:.5f} per {wait_unit}")
        st.write(f"Coefficient of variation = **{sd / mu:.3f}** (an exponential has CV = 1)")
        if sd / mu < 0.6:
            st.warning("CV is far below 1: waiting times are tightly clustered, so the "
                       "exponential (memoryless) model does not describe this variable well.")
        tt = np.linspace(0, max(gaps) * 1.1, 200)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(gaps, bins=15, density=True, alpha=0.6, label="Observed")
        ax.plot(tt, [sc.exp_pdf(x, lam_t) for x in tt], label="Exponential fit")
        ax.set_xlabel(f"{wait_label} ({wait_unit})"); ax.set_ylabel("Density"); ax.legend()
        st.pyplot(fig)
        t = st.slider(f"t ({wait_unit})", float(min(gaps) * 0.5), float(max(gaps) * 1.1), float(mu))
        st.write(f"Model: P(T <= t) = **{sc.exp_cdf(t, lam_t):.4f}**   |   "
                 f"P(T > t) = **{sc.exp_survival(t, lam_t):.4f}**")
        st.write(f"Observed: share of waits <= t = **{np.mean(np.array(gaps) <= t):.4f}**")
    else:
        st.info("Not enough events in this selection for a waiting-time analysis.")

# -------------------------------------------------------------------- step 4
st.header("Step 4 - AI / engineering decision")
status, pk = sc.thermal_risk_decision(lam, int(k_thr), p_thr)
unit = "discharge cycle" if battery_mode else "week"
st.write(f"P(X >= {int(k_thr)} event readings in a {unit}) = **{pk:.4f}**, alert threshold = {p_thr:.2f}")
if status == "HIGH RISK":
    st.error("HIGH RISK -> Battery management system: derate discharge current, raise cooling, "
             "flag the pack for inspection.")
else:
    st.success("NORMAL -> Continue normal operation.")

with st.expander("Limitations"):
    st.markdown("""
- Event readings inside a cycle are **consecutive** (one heating excursion per cycle), so they are not independent events.
- The count depends on the sampling interval (about 9-20 s between readings).
- The event rate depends on battery age, so the constant-rate Poisson assumption holds only within a window.
- This is a single cell (B0005); results may not generalise to other cells or packs.
- `thermal_event` is derived from a 40 C threshold chosen by the group, not an observed failure.
""")
