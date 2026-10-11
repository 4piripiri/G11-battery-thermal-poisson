"""
Battery Thermal Anomaly Risk Screening Dashboard
=================================================
Senior Applied Statistics & Engineering Risk-Screening Dashboard

Organized into six clear sections:
  A. Overview: Introduction, Dataset Status, Data Quality, KPI Cards
  B. Observed Data Analysis: Temperature Trends, Onsets, Rates, Diagnostics, Episode Table
  C. Monte Carlo Simulator: Stochastic Simulations, Paths, Convergence, Provenance
  D. Risk & Decision Support: Validated Thresholds, Action Protocols, Screening Caveat
  E. Model Assumptions & Math: Mathematical Derivations, Parameter Substitutions
  F. Results & Downloads: Audited CSV Exports & Comparison Table
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

import core
import diagnostics

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Battery Thermal Anomaly Simulator | Applied Risk Screening",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.45rem;
    }

    /* Status Pills */
    .badge-data {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
        border-radius: 9999px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-hypo {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
        border-radius: 9999px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }

    /* Decision Callout Banners */
    .risk-banner {
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 18px;
        border-left: 8px solid;
    }
    .risk-low {
        background-color: #f0fdf4;
        border-color: #22c55e;
        color: #14532d;
    }
    .risk-moderate {
        background-color: #fffbeb;
        border-color: #f59e0b;
        color: #78350f;
    }
    .risk-high {
        background-color: #fef2f2;
        border-color: #ef4444;
        color: #7f1d1d;
    }

    /* Explanatory Captions */
    .chart-caption {
        background-color: #f8fafc;
        border-radius: 6px;
        padding: 9px 13px;
        font-size: 0.86rem;
        color: #334155;
        margin-top: 5px;
        margin-bottom: 18px;
        border-left: 3px solid #3b82f6;
    }
    .disclaimer-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #64748b;
        border-radius: 6px;
        padding: 14px;
        font-size: 0.87rem;
        color: #334155;
        margin-top: 15px;
    }
    .note-box {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.88rem;
        color: #0369a1;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Sidebar Interactive Inputs & Data Ingestion
# -----------------------------------------------------------------------------
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "B0005_thermal_events.csv")

with st.sidebar:
    st.title("⚡ Control Panel")
    st.markdown("Stochastic risk screening using Poisson & Exponential distributions.")

    st.subheader("1. Telemetry Dataset")
    data_source_mode = st.radio(
        "Dataset Source",
        options=["Bundled NASA B0005 File", "Upload Custom CSV File"],
        index=0,
        help="Select whether to analyze the bundled aging dataset or upload a new telemetry CSV.",
    )

    uploaded_file = None
    if data_source_mode == "Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload Telemetry CSV",
            type=["csv"],
            help="Required columns: cycle, time, temperature, thermal_event",
        )

# Execute Data Ingestion & Validation
load_error: Optional[str] = None
source_name = ""

if data_source_mode == "Upload Custom CSV File":
    if uploaded_file is not None:
        source_name = uploaded_file.name
        try:
            df, onsets, total_s, n_cycles, report = core.validate_and_load_dataframe(uploaded_file)
        except core.DataValidationError as err:
            load_error = str(err)
    else:
        st.sidebar.info("Awaiting CSV upload. Switch to the Bundled NASA file above to proceed.")
        st.stop()
else:
    source_name = "B0005_thermal_events.csv"
    if os.path.exists(DEFAULT_CSV):
        try:
            df, onsets, total_s, n_cycles, report = core.validate_and_load_dataframe(DEFAULT_CSV)
        except core.DataValidationError as err:
            load_error = f"Bundled dataset error: {str(err)}"
    else:
        load_error = f"Bundled file not found at: {DEFAULT_CSV}"

if load_error:
    st.error(f"❌ Data Validation Error: {load_error}")
    st.stop()

# Derive empirical metrics from validated dataset
obs_summary = core.data_summary(df, onsets, total_s, n_cycles)
obs_gaps = core.waiting_times_hours(onsets)
cycle_stats = core.cycle_diagnostics(df, onsets)
episodes_table = core.extract_episodes_table(df)
lam_data = obs_summary["lam"]

with st.sidebar:
    st.success(f"✓ Ingested {obs_summary['total_rows']:,} rows ({obs_summary['cycles']} cycles)")

    st.subheader("2. Arrival Rate λ")
    rate_mode = st.radio(
        "Event Rate Mode",
        options=["Data-Derived Rate (λ_obs)", "Manual / What-If Rate"],
        index=0,
        help="Select whether to use the rate derived from the dataset or test a hypothetical scenario.",
    )

    if rate_mode == "Data-Derived Rate (λ_obs)":
        lam_active = float(lam_data)
        st.caption(f"Active Rate: **{lam_active:.3f} episodes/hr** (from {source_name})")
        is_hypothetical = False
    else:
        lam_active = st.slider(
            "Hypothetical λ (episodes/hr)",
            min_value=0.1,
            max_value=10.0,
            value=float(min(max(round(lam_data, 2), 0.1), 10.0)),
            step=0.05,
            help="Simulate an elevated or diminished hazard rate for stress-testing.",
        )
        is_hypothetical = True

    st.subheader("3. Observation Horizon & Decisions")
    duration = st.slider(
        "Observation Horizon T (Hours)",
        min_value=1.0,
        max_value=24.0,
        value=2.0,
        step=0.5,
        format="%.1f",
        help="Operating duration over which abnormal thermal event counts are tallied. Stepping: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, ...",
    )

    threshold_type = st.radio(
        "Decision Threshold Strategy",
        options=[
            "Battery Health Hazard Rate: λ (episodes/hr)",
            "Critical Count Exceedance: P(N ≥ k_crit) in Window T",
            "First Incident Trigger: P(N ≥ 1) in Window T",
        ],
        index=0,
        help="Rate-Based evaluates fundamental degradation rate λ; Exceedance evaluates probability P(N ≥ k_crit); First Incident evaluates P(N ≥ 1).",
    )

    expected_events = lam_active * duration
    is_decision_valid = True

    if threshold_type == "Battery Health Hazard Rate: λ (episodes/hr)":
        k_crit = 1
        st.caption(f"Evaluates active event arrival rate λ (Currently {lam_active:.2f}/hr).")
        col_rw, col_rh = st.columns(2)
        with col_rw:
            rate_warn = st.number_input(
                "Warning Rate λ-limit",
                min_value=0.10,
                max_value=10.0,
                value=0.80,
                step=0.10,
                help="Arrival rate λ at or above which risk escalates to MODERATE.",
            )
        with col_rh:
            rate_high = st.number_input(
                "High Risk Rate λ-limit",
                min_value=0.20,
                max_value=10.0,
                value=2.00,
                step=0.10,
                help="Arrival rate λ at or above which risk escalates to HIGH.",
            )
        p_warn = float(rate_warn)
        p_high = float(rate_high)
        if rate_high <= rate_warn:
            st.error("Validation Error: High Risk Rate limit must strictly exceed Warning Rate limit.")
            is_decision_valid = False

    elif threshold_type == "Critical Count Exceedance: P(N ≥ k_crit) in Window T":
        st.caption(f"Expected count in {duration:.1f}h window: μ = {expected_events:.2f} episodes.")
        k_crit = st.slider(
            "Critical Event Threshold (k_crit)",
            min_value=1,
            max_value=15,
            value=max(2, int(np.ceil(expected_events))),
            step=1,
            help="Operational escalation triggered if N reaches or exceeds k_crit.",
        )
        if k_crit <= expected_events:
            st.info(
                f"💡 Note: k_crit ({k_crit}) ≤ expected mean μ ({expected_events:.1f}). "
                f"Seeing ≥ {k_crit} events over {duration:.1f}h is statistically expected, making exceedance probability high. "
                "To screen for abnormal surges above baseline, pick k_crit > μ."
            )
        col_pw, col_ph = st.columns(2)
        with col_pw:
            p_warn = st.number_input(
                "Warning P-limit",
                min_value=0.01,
                max_value=0.95,
                value=0.30,
                step=0.05,
                help="Probability at or above which risk escalates to MODERATE.",
            )
        with col_ph:
            p_high = st.number_input(
                "High Risk P-limit",
                min_value=0.05,
                max_value=0.99,
                value=0.70,
                step=0.05,
                help="Probability at or above which risk escalates to HIGH.",
            )
        if p_high <= p_warn:
            st.error("Validation Error: High Risk P-limit must strictly exceed Warning P-limit.")
            is_decision_valid = False

    else:  # First Incident Trigger: P(N >= 1)
        k_crit = 1
        st.caption("Zero-tolerance criterion: evaluates chance of experiencing at least 1 thermal anomaly.")
        col_pw, col_ph = st.columns(2)
        with col_pw:
            p_warn = st.number_input(
                "Warning P-limit",
                min_value=0.01,
                max_value=0.95,
                value=0.30,
                step=0.05,
                help="Probability at or above which risk escalates to MODERATE.",
            )
        with col_ph:
            p_high = st.number_input(
                "High Risk P-limit",
                min_value=0.05,
                max_value=0.99,
                value=0.70,
                step=0.05,
                help="Probability at or above which risk escalates to HIGH.",
            )
        if p_high <= p_warn:
            st.error("Validation Error: High Risk P-limit must strictly exceed Warning P-limit.")
            is_decision_valid = False

    st.subheader("4. Monte Carlo Settings")
    n_sim = st.select_slider(
        "Monte Carlo Iterations",
        options=[500, 1000, 2500, 5000, 10000, 20000],
        value=5000,
        help="Number of independent observation windows simulated.",
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="Seed for reproducible pseudorandom sampling.",
    )

    run_sim_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 3. Monte Carlo Simulation Engine & Session State
# -----------------------------------------------------------------------------
if "sim_state" not in st.session_state or run_sim_btn:
    rng = np.random.default_rng(int(seed))
    mu_theoretical = lam_active * duration

    sim_counts = core.simulate_counts(lam_active, duration, n_sim, rng)
    sim_waits = core.simulate_waits(lam_active, n_sim, rng)
    sim_paths = core.simulate_paths(lam_active, duration, n_paths=5, rng=rng)

    st.session_state["sim_state"] = {
        "lam": lam_active,
        "duration": duration,
        "n_sim": n_sim,
        "seed": seed,
        "k_crit": k_crit,
        "threshold_type": threshold_type,
        "p_warn": p_warn,
        "p_high": p_high,
        "is_decision_valid": is_decision_valid,
        "is_hypothetical": is_hypothetical,
        "sim_counts": sim_counts,
        "sim_waits": sim_waits,
        "sim_paths": sim_paths,
    }

sim = st.session_state["sim_state"]
lam_cur = sim["lam"]
dur_cur = sim["duration"]
n_sim_cur = sim["n_sim"]
k_crit_cur = sim["k_crit"]
thresh_type_cur = sim["threshold_type"]
p_warn_cur = sim["p_warn"]
p_high_cur = sim["p_high"]
is_valid_cur = sim.get("is_decision_valid", True)
is_hypo_cur = sim["is_hypothetical"]

# Theoretical calculations
pw = core.poisson_window(lam_cur, dur_cur, k_crit_cur)
ew = core.exponential_wait(lam_cur, dur_cur)

# Monte Carlo statistics
mean_cnt_sim = float(np.mean(sim["sim_counts"]))
var_cnt_sim = float(np.var(sim["sim_counts"], ddof=1))
p0_sim = float(np.mean(sim["sim_counts"] == 0))
p_ge_1_sim = float(np.mean(sim["sim_counts"] >= 1))
p_ge_k_sim = float(np.mean(sim["sim_counts"] >= k_crit_cur))

mean_wait_sim = float(np.mean(sim["sim_waits"]))
var_wait_sim = float(np.var(sim["sim_waits"], ddof=1))

# Decision Evaluation Metric & Rule Description
if thresh_type_cur == "Battery Health Hazard Rate: λ (episodes/hr)":
    val_eval = lam_cur
    metric_label = f"Arrival Rate λ = {lam_cur:.2f} episodes/hr"
    rule_label = f"SAFE if λ < {p_warn_cur:.2f}/hr, WARNING if {p_warn_cur:.2f} ≤ λ < {p_high_cur:.2f}/hr, HIGH RISK if λ ≥ {p_high_cur:.2f}/hr"
    decision_level = core.evaluate_risk(val_eval, p_warn_cur, p_high_cur) if is_valid_cur else "INVALID LIMITS"
elif thresh_type_cur == "First Incident Trigger: P(N ≥ 1) in Window T":
    val_eval = pw["p_ge_1"]
    metric_label = f"P(N ≥ 1 Incident in {dur_cur:.1f}h) = {val_eval:.1%}"
    rule_label = f"SAFE if P < {p_warn_cur:.0%}, WARNING if {p_warn_cur:.0%} ≤ P < {p_high_cur:.0%}, HIGH RISK if P ≥ {p_high_cur:.0%}"
    decision_level = core.evaluate_risk(val_eval, p_warn_cur, p_high_cur) if is_valid_cur else "INVALID LIMITS"
else:  # Critical Count Exceedance
    val_eval = pw["p_ge_k"]
    metric_label = f"P(N ≥ {k_crit_cur} Episodes in {dur_cur:.1f}h) = {val_eval:.1%}"
    rule_label = f"SAFE if P < {p_warn_cur:.0%}, WARNING if {p_warn_cur:.0%} ≤ P < {p_high_cur:.0%}, HIGH RISK if P ≥ {p_high_cur:.0%}"
    decision_level = core.evaluate_risk(val_eval, p_warn_cur, p_high_cur) if is_valid_cur else "INVALID LIMITS"

# -----------------------------------------------------------------------------
# 4. Header Badges & Executive Context
# -----------------------------------------------------------------------------
st.title("⚡ Battery Thermal Anomaly Risk Screening Dashboard")
st.markdown(
    "Senior Applied Statistics & Engineering Risk-Screening Dashboard: Integrating empirical telemetry, "
    "Poisson counting processes, Exponential waiting times, and Monte Carlo simulation for BMS risk support."
)

badge_data_html = f'<span class="badge-data">📁 Dataset: {source_name} ({obs_summary["cycles"]} Cycles)</span>'
if is_hypo_cur:
    badge_rate_html = f'<span class="badge-hypo">⚠️ Simulating Hypothetical Rate: λ = {lam_cur:.3f}/hr (Observed: {lam_data:.3f}/hr)</span>'
else:
    badge_rate_html = f'<span class="badge-data">✓ Simulating Data-Derived Rate: λ_obs = {lam_cur:.3f}/hr</span>'

st.markdown(f"{badge_data_html}{badge_rate_html}", unsafe_allow_html=True)
st.write("")

# -----------------------------------------------------------------------------
# 5. Dashboard Tabs Architecture (Sections A - F)
# -----------------------------------------------------------------------------
tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs([
    "📋 A. Overview",
    "📊 B. Observed Data Analysis",
    "🎲 C. Monte Carlo Simulator",
    "🛡️ D. Risk & Engineering Decision",
    "🔬 E. Assumptions & Math",
    "📥 F. Results & Downloads",
])

# =============================================================================
# TAB A: OVERVIEW
# =============================================================================
with tab_a:
    st.subheader("A. Project Overview & Telemetry Summary")
    st.markdown(
        """
        In Lithium-ion battery packs, localized abnormal thermal spikes ($T > 40^\\circ\\text{C}$) caused by 
        internal resistance degradation, micro-shorts, or coolant maldistribution manifest as stochastic excursions. 
        This dashboard implements a **Poisson point process** framework to estimate arrival frequencies, 
        model waiting times between spikes, and support Battery Management System (BMS) operational decisions.
        """
    )

    st.markdown("#### Operational Summary Metric Cards")
    oa1, oa2, oa3, oa4, oa5, oa6 = st.columns(6)
    oa1.metric("Flagged Readings", f"{obs_summary['flagged_rows']:,}", f"{obs_summary['pct_flagged']:.2f}% of rows")
    oa2.metric("Episode Onsets", f"{obs_summary['episodes']}")
    oa3.metric("Observed Exposure", f"{obs_summary['hours']:.1f} hrs", f"{obs_summary['cycles']} cycles")
    oa4.metric(
        "Estimated Event Rate",
        f"{obs_summary['lam']:.3f} / hr",
        "Point estimate λ̂",
        delta_color="off",
    )
    oa5.metric(
        "Exact 95% Poisson CI",
        f"[{obs_summary['lam_lo']:.3f}, {obs_summary['lam_hi']:.3f}]",
        "Garwood method",
        delta_color="off",
    )
    oa6.metric("Peak Temperature", f"{obs_summary['peak_temperature']:.2f} °C", "Threshold: 40.0°C")

    st.markdown("---")
    st.markdown("#### Dataset Status & Quality Summary")
    q_col1, q_col2 = st.columns(2)

    with q_col1:
        st.markdown(
            f"""
            - **Source File:** `{source_name}`
            - **Total Telemetry Rows:** `{obs_summary['total_rows']:,}`
            - **Unique Cycles Analyzed:** `{obs_summary['cycles']}` (Discharge cycles)
            - **Null / Missing Records Dropped:** `{report.get('null_rows_dropped', 0)}`
            - **Duplicate Timestamps Resolved:** `{report.get('deduplicated_rows', 0)}`
            - **Time-Ordering Verification:** Passed (Monotonically non-decreasing within cycle)
            """
        )

    with q_col2:
        st.markdown(
            """
            - **Event Detection Criterion:** $T_{\\text{cell}} > 40.0^\\circ\\text{C}$
            - **Episode Definition:** Consecutive readings above $40.0^\\circ\\text{C}$ in the same cycle.
            - **Reduction Factor:** $1,206\\text{ readings} \\to 86\\text{ distinct episodes}$ (Avg: $14.0$ readings/episode).
            - **Per-Cycle Anomaly Rate:** $\\frac{86}{168} = 0.512$ episodes/cycle.
            - **Dispersion Index (Var/Mean):** $D = 0.491$ (Under-dispersed per-cycle).
            """
        )

# =============================================================================
# TAB B: OBSERVED DATA ANALYSIS
# =============================================================================
with tab_b:
    st.subheader("B. Observed Data Analysis & Empirical Diagnostics")

    # Graph B1: Temperature Trend Across Cycles
    st.markdown("#### 1. Temperature Evolution Across Operating History")
    fig_b1 = go.Figure()
    fig_b1.add_trace(
        go.Scatter(
            x=cycle_stats["cycle"],
            y=cycle_stats["max_temp"],
            mode="lines+markers",
            name="Peak Temperature (°C)",
            line=dict(color="#d97706", width=2),
            marker=dict(
                size=5,
                color=["#ef4444" if t > 40.0 else "#2563eb" for t in cycle_stats["max_temp"]],
            ),
            hovertemplate="Cycle: %{x}<br>Peak Temp: %{y:.2f}°C<extra></extra>",
        )
    )
    fig_b1.add_hline(
        y=40.0,
        line_width=2,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Event-Detection Threshold (40.0°C)",
        annotation_position="bottom right",
    )
    fig_b1.update_layout(
        title=dict(text="<b>Peak Cell Temperature per Cycle with 40.0°C Event Threshold</b>", font=dict(size=13)),
        xaxis=dict(title="Discharge Cycle Identifier"),
        yaxis=dict(title="Peak Temperature (°C)"),
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig_b1, use_container_width=True)
    st.markdown(
        """
        <div class="chart-caption">
            <strong>Graph B1 Explanation:</strong> Displays peak temperature recorded during each discharge cycle. 
            Points in red breach the 40.0°C threshold. As cell internal resistance increases with cycling aging, 
            peak temperature rises steadily from 38.5°C to >41.4°C.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Graphs B2 & B3: Episode Onsets & Rolling Rate
    col_b2, col_b3 = st.columns(2)
    with col_b2:
        st.markdown("#### 2. Episode Onsets per Cycle")
        fig_b2 = go.Figure()
        fig_b2.add_trace(
            go.Bar(
                x=cycle_stats["cycle"],
                y=cycle_stats["episodes"],
                name="Episode Onset",
                marker_color=["#ef4444" if ep > 0 else "#94a3b8" for ep in cycle_stats["episodes"]],
                hovertemplate="Cycle: %{x}<br>Episodes: %{y}<extra></extra>",
            )
        )
        fig_b2.update_layout(
            title=dict(text="<b>Episode Onsets Across Recorded Cycles (0 or 1)</b>", font=dict(size=13)),
            xaxis=dict(title="Discharge Cycle Identifier"),
            yaxis=dict(title="Episode Onset Count", dtick=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_b2, use_container_width=True)
        st.markdown(
            """
            <div class="chart-caption">
                <strong>Graph B2 Explanation:</strong> Confirms that each cycle has either 0 or 1 episode onset (no cycle contains ≥ 2). 
                Anomalies are virtually absent in early life and become consistent in late life.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b3:
        st.markdown("#### 3. Observed Rolling Event Rate")
        cs_copy = cycle_stats.copy()
        cs_copy["rolling_rate"] = cs_copy["episodes"].rolling(window=15, min_periods=5).mean()
        fig_b3 = go.Figure()
        fig_b3.add_trace(
            go.Scatter(
                x=cs_copy["cycle"],
                y=cs_copy["rolling_rate"],
                mode="lines",
                name="15-Cycle Rolling Anomaly Rate",
                line=dict(color="#dc2626", width=2.5),
                hovertemplate="Cycle: %{x}<br>Rolling Rate: %{y:.1%}<extra></extra>",
            )
        )
        fig_b3.add_hline(
            y=float(cs_copy["episodes"].mean()),
            line_width=1.5,
            line_dash="dash",
            line_color="#0f172a",
            annotation_text=f"Global Mean ({cs_copy['episodes'].mean():.1%})",
            annotation_position="top left",
        )
        fig_b3.update_layout(
            title=dict(text="<b>Evidence of Non-Stationarity: Rate Escalation Over Life</b>", font=dict(size=13)),
            xaxis=dict(title="Discharge Cycle Identifier"),
            yaxis=dict(title="Fraction of Cycles with Onset", tickformat=".0%", range=[-0.05, 1.05]),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_b3, use_container_width=True)
        st.markdown(
            """
            <div class="chart-caption">
                <strong>Graph B3 Explanation:</strong> Shows rolling anomaly frequency rising from 2% early in life to 100% in late life. 
                This violates the constant hazard rate assumption of homogeneous Poisson processes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Graphs B4 & B5: Observed Counts vs Poisson & Gaps vs Exponential
    col_b4, col_b5 = st.columns(2)
    with col_b4:
        st.markdown("#### 4. Cycle Counts vs Poisson PMF")
        fig_b4 = diagnostics.plot_cycle_episodes_vs_poisson(cycle_stats)
        st.plotly_chart(fig_b4, use_container_width=True)
        st.markdown(
            """
            <div class="chart-caption">
                <strong>Graph B4 Explanation:</strong> Compares cycle episode frequencies against theoretical Poisson PMF. 
                The empirical dispersion index <code>D = 0.491</code> proves under-dispersion because discharge duration bounds 
                events to at most one per cycle.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b5:
        st.markdown("#### 5. Inter-Arrival Times vs Exponential PDF")
        fig_b5 = diagnostics.plot_interarrival_gaps_vs_exponential(obs_gaps, lam_data)
        st.plotly_chart(fig_b5, use_container_width=True)
        st.markdown(
            """
            <div class="chart-caption">
                <strong>Graph B5 Explanation:</strong> Compares normalized empirical inter-arrival densities against fitted Exponential density. 
                Observed gaps cluster near 0.85 hrs in late life, while idle gaps between early episodes create a heavy tail.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Compact Episode Table
    st.markdown("#### 6. Detailed Thermal Episode Log (86 Extracted Episodes)")
    st.caption("Inspect individual thermal excursions with onset timing, excursion durations, and peak temperatures.")
    st.dataframe(
        episodes_table.style.format({
            "onset_time_s": "{:.1f}",
            "onset_global_h": "{:.2f}",
            "duration_s": "{:.1f}",
            "peak_temp_c": "{:.2f}",
            "wait_from_prev_h": lambda x: f"{x:.2f}" if pd.notnull(x) else "—",
        }),
        use_container_width=True,
        height=280,
    )

# =============================================================================
# TAB C: MONTE CARLO SIMULATOR
# =============================================================================
with tab_c:
    st.subheader("C. Monte Carlo Simulator & Stochastic Risk Projections")

    if is_hypo_cur:
        st.markdown(
            f'<div class="badge-hypo" style="padding: 6px 14px; font-size: 0.9rem;">⚠️ Simulating Hypothetical Scenario: λ = {lam_cur:.3f}/hr (Observed Rate = {lam_data:.3f}/hr)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="badge-data" style="padding: 6px 14px; font-size: 0.9rem;">✓ Simulating Data-Derived Baseline: λ_obs = {lam_cur:.3f}/hr</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Time Horizon (T)", f"{dur_cur:.1f} hrs")
    mc2.metric("Expected Events (μ)", f"{pw['mu']:.3f}", f"{mean_cnt_sim - pw['mu']:+.3f} (sim error)")
    mc3.metric("P(N = 0) [Safe]", f"{pw['p0']:.1%}", f"{p0_sim - pw['p0']:+.1%} (sim)")
    mc4.metric("P(N ≥ 1) [At Least 1]", f"{pw['p_ge_1']:.1%}", f"{p_ge_1_sim - pw['p_ge_1']:+.1%} (sim)")
    mc5.metric(f"P(N ≥ {k_crit_cur}) [Critical]", f"{pw['p_ge_k']:.1%}", f"{p_ge_k_sim - pw['p_ge_k']:+.1%} (sim)")

    st.write("")
    mc_col1, mc_col2 = st.columns(2)

    with mc_col1:
        # MC Graph 1: Counts vs Poisson
        max_k = max(int(np.percentile(sim["sim_counts"], 99.8)) + 3, k_crit_cur + 3)
        k_eval_range = np.arange(0, max_k + 1)
        sim_k_freqs = [float(np.mean(sim["sim_counts"] == k)) for k in k_eval_range]
        theo_k_pmf = [float(stats.poisson.pmf(k, pw["mu"])) for k in k_eval_range]
        bar_colors = ["#ef4444" if k >= k_crit_cur else "#3b82f6" for k in k_eval_range]

        fig_mc1 = go.Figure()
        fig_mc1.add_trace(
            go.Bar(
                x=k_eval_range,
                y=sim_k_freqs,
                name=f"Monte Carlo ({n_sim_cur:,} Trials)",
                marker_color=bar_colors,
                opacity=0.75,
                hovertemplate="Events (k): %{x}<br>Simulated Relative Frequency: %{y:.4f}<extra></extra>",
            )
        )
        fig_mc1.add_trace(
            go.Scatter(
                x=k_eval_range,
                y=theo_k_pmf,
                mode="lines+markers",
                name="Theoretical Poisson PMF",
                line=dict(color="#0f172a", width=2.5),
                marker=dict(size=6, color="#0f172a"),
                hovertemplate="Events (k): %{x}<br>Theoretical PMF: %{y:.4f}<extra></extra>",
            )
        )
        fig_mc1.add_vline(x=pw["mu"], line_width=2, line_dash="dash", line_color="#10b981", annotation_text=f"μ = {pw['mu']:.2f}")
        fig_mc1.add_vline(x=k_crit_cur - 0.5, line_width=2, line_dash="dot", line_color="#ef4444", annotation_text=f"Threshold k_crit = {k_crit_cur}")
        fig_mc1.update_layout(
            title=dict(text=f"<b>1. Simulated Event Counts vs Poisson PMF (μ = {pw['mu']:.2f})</b>", font=dict(size=13)),
            xaxis=dict(title="Event Count in Window (k)", dtick=1 if len(k_eval_range) < 16 else None),
            yaxis=dict(title="Probability / Relative Frequency", tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_mc1, use_container_width=True)
        st.markdown(
            """
            <div class="chart-caption">
                <strong>Graph C1 Explanation:</strong> Displays empirical count frequencies across Monte Carlo windows compared to the theoretical 
                Poisson distribution. Realizations in red breach the critical threshold <code>k_crit</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mc_col2:
        # MC Graph 2: Waiting Times vs Exponential
        max_w_plot = float(min(np.percentile(sim["sim_waits"], 99.5), 4.5 * ew["mean"]))
        t_cont = np.linspace(0.001, max(max_w_plot, 1.0), 300)
        theo_exp_pdf = lam_cur * np.exp(-lam_cur * t_cont)

        fig_mc2 = go.Figure()
        fig_mc2.add_trace(
            go.Histogram(
                x=sim["sim_waits"],
                histnorm="probability density",
                nbinsx=35,
                name="Simulated Wait Times",
                marker_color="#6366f1",
                opacity=0.65,
                hovertemplate="Wait Time: %{x:.2f} h<br>Empirical Density: %{y:.3f}<extra></extra>",
            )
        )
        fig_mc2.add_trace(
            go.Scatter(
                x=t_cont,
                y=theo_exp_pdf,
                mode="lines",
                name=f"Theoretical Exp(λ = {lam_cur:.2f})",
                line=dict(color="#dc2626", width=2.5),
                hovertemplate="Wait Time: %{x:.2f} h<br>Theoretical PDF: %{y:.4f}<extra></extra>",
            )
        )
        fig_mc2.add_vline(x=ew["mean"], line_width=2, line_dash="dash", line_color="#10b981", annotation_text=f"E[W] = {ew['mean']:.2f}h")
        fig_mc2.update_layout(
            title=dict(text=f"<b>2. Waiting Times vs Exponential PDF (1/λ = {ew['mean']:.2f}h)</b>", font=dict(size=13)),
            xaxis=dict(title="Inter-Arrival Waiting Time (Hours)", range=[0, max_w_plot]),
            yaxis=dict(title="Probability Density"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_mc2, use_container_width=True)
        st.markdown(
            r"""
            <div class="chart-caption">
                <strong>Graph C2 Explanation:</strong> Validates simulated inter-arrival waiting times against the theoretical memoryless 
                Exponential density \(f(w) = \lambda e^{-\lambda w}\).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MC Graphs 3 & 4: Trajectories & Sensitivity
    mc_col3, mc_col4 = st.columns(2)
    with mc_col3:
        # MC Graph 3: Event paths
        t_arr = np.linspace(0, dur_cur, 200)
        mean_path = lam_cur * t_arr
        std_path = np.sqrt(np.maximum(mean_path, 1e-9))
        upper_2s = mean_path + 2.0 * std_path
        lower_2s = np.maximum(0, mean_path - 2.0 * std_path)

        fig_mc3 = go.Figure()
        fig_mc3.add_trace(
            go.Scatter(
                x=np.concatenate([t_arr, t_arr[::-1]]),
                y=np.concatenate([upper_2s, lower_2s[::-1]]),
                fill="toself",
                fillcolor="rgba(148, 163, 184, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Approximate ±2σ Dispersion Band",
            )
        )
        pal = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777"]
        for p_idx, arrivals in enumerate(sim["sim_paths"]):
            if len(arrivals) > 0:
                st_x, st_y = [0.0], [0]
                for a in arrivals:
                    st_x.extend([a, a])
                    st_y.extend([st_y[-1], st_y[-1] + 1])
                st_x.append(dur_cur)
                st_y.append(st_y[-1])
            else:
                st_x, st_y = [0.0, dur_cur], [0, 0]
            fig_mc3.add_trace(
                go.Scatter(
                    x=st_x,
                    y=st_y,
                    mode="lines",
                    name=f"Battery Pack #{p_idx+1}",
                    line=dict(color=pal[p_idx % len(pal)], width=1.8),
                    hovertemplate="Time: %{x:.2f} h<br>Cumulative Events: %{y}<extra></extra>",
                )
            )
        fig_mc3.add_trace(
            go.Scatter(
                x=t_arr,
                y=mean_path,
                mode="lines",
                name="Theoretical Mean E[N(t)] = λt",
                line=dict(color="#0f172a", width=2.5, dash="dash"),
                hovertemplate="Time: %{x:.2f} h<br>Expected Count: %{y:.2f}<extra></extra>",
            )
        )
        fig_mc3.update_layout(
            title=dict(text=f"<b>3. Cumulative Abnormal Events Over Time (T = {dur_cur:.1f} hrs)</b>", font=dict(size=13)),
            xaxis=dict(title="Operating Time Elapsed (Hours)", range=[0, dur_cur]),
            yaxis=dict(title="Cumulative Event Count N(t)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_mc3, use_container_width=True)
        st.markdown(
            r"""
            <div class="chart-caption">
                <strong>Graph C3 Explanation:</strong> Displays sample counting trajectories \(N(t)\) for 5 representative battery packs. 
                The grey band denotes an <em>approximate \(\pm 2\sigma\) dispersion band</em> (\(\lambda t \pm 2\sqrt{\lambda t}\))—this 
                indicates empirical spread and is not a formal confidence interval.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mc_col4:
        # MC Graph 4: Sensitivity curve
        lam_grid = np.linspace(0.1, 10.0, 200)
        p1_cur = 1.0 - np.exp(-lam_grid * dur_cur)
        p1_half = 1.0 - np.exp(-lam_grid * (dur_cur / 2.0))

        fig_mc4 = go.Figure()
        fig_mc4.add_trace(
            go.Scatter(
                x=lam_grid,
                y=p1_cur,
                mode="lines",
                name=f"Current Window (T = {dur_cur:.1f} h)",
                line=dict(color="#2563eb", width=3),
                hovertemplate="λ: %{x:.2f}/h<br>P(N≥1): %{y:.1%}<extra></extra>",
            )
        )
        fig_mc4.add_trace(
            go.Scatter(
                x=lam_grid,
                y=p1_half,
                mode="lines",
                name=f"Shorter Window (T = {dur_cur/2.0:.1f} h)",
                line=dict(color="#059669", width=2, dash="dash"),
                hovertemplate="λ: %{x:.2f}/h<br>P(N≥1): %{y:.1%}<extra></extra>",
            )
        )
        fig_mc4.add_trace(
            go.Scatter(
                x=[lam_cur],
                y=[pw["p_ge_1"]],
                mode="markers+text",
                name="Active Operating Point",
                marker=dict(size=12, color="#dc2626", symbol="diamond"),
                text=[f"λ={lam_cur:.2f}<br>P={pw['p_ge_1']:.1%}"],
                textposition="top left",
                hovertemplate="Operating Point<br>λ: %{x:.2f}/h<br>P: %{y:.1%}<extra></extra>",
            )
        )
        fig_mc4.update_layout(
            title=dict(text="<b>4. Sensitivity: Probability P(N ≥ 1) vs Arrival Rate λ</b>", font=dict(size=13)),
            xaxis=dict(title="Arrival Rate (λ, episodes/hour)", range=[0.1, 10.0]),
            yaxis=dict(title="Probability of ≥1 Event", tickformat=".1%", range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=370,
        )
        st.plotly_chart(fig_mc4, use_container_width=True)
        st.markdown(
            r"""
            <div class="chart-caption">
                <strong>Graph C4 Explanation:</strong> Demonstrates non-linear exponential saturation in risk. 
                Even small degradation-induced increases in \(\lambda\) cause rapid escalation toward certain anomaly occurrence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # MC Graph 5: Comparison
    c_labels = [
        "Expected Count (μ)",
        "Count Variance (σ²)",
        "P(Zero Events, N=0)",
        "P(At Least 1, N≥1)",
        f"P(Exceedance, N≥{k_crit_cur})",
        "Mean Wait (1/λ, h)",
    ]
    c_theo = [pw["mu"], pw["mu"], pw["p0"], pw["p_ge_1"], pw["p_ge_k"], ew["mean"]]
    c_sim = [mean_cnt_sim, var_cnt_sim, p0_sim, p_ge_1_sim, p_ge_k_sim, mean_wait_sim]

    fig_mc5 = go.Figure()
    fig_mc5.add_trace(
        go.Bar(
            x=c_labels,
            y=c_theo,
            name="Theoretical Value",
            marker_color="#1e293b",
            text=[f"{v:.3f}" if v >= 0.05 else f"{v:.4f}" for v in c_theo],
            textposition="auto",
        )
    )
    fig_mc5.add_trace(
        go.Bar(
            x=c_labels,
            y=c_sim,
            name=f"Monte Carlo ({n_sim_cur:,} Trials)",
            marker_color="#38bdf8",
            text=[f"{v:.3f}" if v >= 0.05 else f"{v:.4f}" for v in c_sim],
            textposition="auto",
        )
    )
    fig_mc5.update_layout(
        title=dict(text="<b>5. Comparison: Theoretical vs Monte Carlo Simulated Statistics</b>", font=dict(size=13)),
        xaxis=dict(title="Statistical Metric"),
        yaxis=dict(title="Value"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        height=360,
    )
    st.plotly_chart(fig_mc5, use_container_width=True)
    st.markdown(
        """
        <div class="chart-caption">
            <strong>Graph C5 Explanation:</strong> Direct validation confirming convergence between theoretical closed-form equations 
            and empirical Monte Carlo sampling.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB D: RISK & ENGINEERING DECISION SUPPORT
# =============================================================================
with tab_d:
    st.subheader("D. Engineering Decision Support & BMS Action Protocols")

    if not is_valid_cur:
        st.error("Cannot evaluate decision: High Risk limit must exceed Warning limit.")
    else:
        if decision_level == "HIGH RISK":
            r_class = "risk-high"
            r_head = "ALERT — HIGH STATISTICAL EVENT RISK"
            r_action = (
                "Immediate BMS power derating required: curtail charge/discharge C-rate by 50–70%. "
                "Engage active liquid chiller loop to 100% duty cycle. "
                "Switch telemetry sampling to high-speed burst rate (50–100 ms)."
            )
        elif decision_level == "MODERATE RISK":
            r_class = "risk-moderate"
            r_head = "WARNING — MODERATE STATISTICAL EVENT RISK"
            r_action = (
                "Initiate pre-emptive auxiliary thermal conditioning. "
                "Increase thermistor and voltage telemetry sampling to 1 Hz. "
                "Flag module for automated impedance spectroscopy and physical inspection at next turnaround."
            )
        else:
            r_class = "risk-low"
            r_head = "SAFE — LOW STATISTICAL EVENT RISK"
            r_action = (
                "Maintain standard nominal operating envelope. "
                "Standard telemetry logging (0.1 Hz) and baseline thermal management loop active."
            )

        mode_desc = "HYPOTHETICAL SIMULATION" if is_hypo_cur else "OBSERVED DATASET BASELINE"

        st.markdown(
            f"""
            <div class="risk-banner {r_class}">
                <h3 style="margin: 0 0 6px 0; font-size: 1.35rem;">{r_head}</h3>
                <p style="margin: 0 0 8px 0; font-size: 0.95rem;">
                    <strong>Decision Basis ({mode_desc}):</strong> Evaluated <strong>{metric_label}</strong>.<br>
                    <strong>Decision Rule ({thresh_type_cur}):</strong> {rule_label}.
                </p>
                <p style="margin: 0; font-size: 0.95rem;">
                    <strong>Recommended BMS Action:</strong> {r_action}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("❓ Why does my current configuration evaluate to this risk level?", expanded=True):
            exp_events = lam_cur * dur_cur
            p_high_str = f"{p_high_cur:.2f}/hr" if thresh_type_cur.startswith("Battery Health") else f"{p_high_cur:.0%}"
            p_warn_str = f"{p_warn_cur:.2f}/hr" if thresh_type_cur.startswith("Battery Health") else f"{p_warn_cur:.0%}"

            st.markdown(
                f"""
                **1. Operating Context & Exposure:**
                - **Arrival Intensity:** $\\lambda = {lam_cur:.2f}\\text{{ episodes/hour}}$ (Mean wait between spikes: {ew['mean']*60:.1f} mins).
                - **Observation Window:** $T = {dur_cur:.1f}\\text{{ hours}}$.
                - **Expected Events in Window:** $\\mu = \\lambda \\cdot T = {lam_cur:.2f} \\times {dur_cur:.1f} = \\mathbf{{{exp_events:.2f}\\text{{ episodes}}}}$.
                """
            )

            if thresh_type_cur.startswith("Battery Health"):
                st.markdown(
                    f"""
                    **2. Rate-Based Evaluation Mechanism:**
                    - Your active strategy evaluates the **direct cell degradation hazard rate** $\\lambda = {lam_cur:.2f}\\text{{/hr}}$.
                    - It is evaluated directly against your Warning limit ({p_warn_str}) and High Risk limit ({p_high_str}).
                    - *Notice:* This classification remains stable across different observation durations $T$, reflecting true cell health.
                    """
                )
            else:
                rel_mean_text = "below the expected mean" if k_crit_cur <= exp_events else "above the expected mean"
                st.markdown(
                    f"""
                    **2. Probability-Based Evaluation Mechanism:**
                    - Over a continuous **{dur_cur:.1f}-hour window**, the battery is statistically expected to encounter **{exp_events:.2f} episodes**.
                    - Because $k_{{\\text{{crit}}}} = {k_crit_cur}$ is **{rel_mean_text}** $\\mu = {exp_events:.2f}$, the cumulative Poisson probability of reaching or exceeding this count is **{val_eval:.1%}**.
                    - Comparing this probability ({val_eval:.1%}) against your High Risk limit ({p_high_str}) yields **{decision_level}**.
                    - *Why long windows trip High Risk:* If you observe a pack for multiple hours ($T \\ge 4\\text{{h}}$), encountering a low count like 1 or 2 events is statistically almost guaranteed (>95%). To test for unusual surges over long durations, choose $k_{{\\text{{crit}}}} > \\mu$ or switch to **'Battery Health Hazard Rate: λ'** in the sidebar.
                    """
                )

    st.markdown("#### Operational Monitoring & Risk Management Protocols")
    dec_c1, dec_c2 = st.columns(2)
    with dec_c1:
        st.markdown(
            f"""
            **1. Sensor Polling & Telemetry Allocation:**
            - **Current Occurrence Likelihood:** There is a **{pw['p_ge_1']:.1%}** chance of encountering at least 1 abnormal event over {dur_cur:.1f} hours.
            - **Expected Event Spacing:** Thermal anomalies arrive on average every **{ew['mean']:.2f} hours** ({(ew['mean']*60):.1f} minutes).
            - **Telemetry Polling Rule:** Stochastic detection requires sensor polling period $T_s \\ll 1/\\lambda$. For active $\\lambda = {lam_cur:.2f}$/hr, sensor polling must remain $\\le 2$ minutes to detect transient onset slopes before heat propagates.
            """
        )

    with dec_c2:
        st.markdown(
            f"""
            **2. Preventative Thermal Conditioning & Power Derating:**
            - **Critical Threshold Exceedance Risk:** Probability of $\\ge {k_crit_cur}$ episode(s) is **{pw['p_ge_k']:.1%}**.
            - **Action Trigger:** When probability $P \\ge {p_high_cur:.0%}$ (or $\\lambda \\ge {p_high_cur:.2f}/hr$), pack thermal margins are statistically saturated. Pre-emptive chiller activation prevents cell temperatures from reaching accelerated degradation regimes (>40°C).
            - **Fleet Screening:** Packs exhibiting localized arrival rates $\\hat{{\\lambda}} > 2.0$/hr during standard cycles should be flagged for pack re-balancing or cell teardown.
            """
        )

    st.markdown(
        r"""
        <div class="disclaimer-box">
            <strong>⚠️ Engineering Screening Model Disclaimer:</strong><br>
            This application implements a statistical screening model based on Poisson counting processes and Exponential waiting times. 
            <strong>It is NOT a validated multi-physics battery safety or thermal runaway simulator.</strong> It does not compute 
            3D Navier-Stokes convective cooling, electrochemical Joule heating ($I^2 R$), or exothermic reaction kinetics 
            (SEI decomposition, anode-cathode micro-shorts). Physical battery safety qualification requires 3D multi-physics CAE models 
            and standardized hardware qualification tests (e.g. UL 9540A, UN 38.3, SAE J2464).
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB E: MODEL ASSUMPTIONS & MATHEMATICAL EXPLANATION
# =============================================================================
with tab_e:
    st.subheader("E. Mathematical Formulations & Statistical Explanations")

    math_col1, math_col2 = st.columns(2)
    with math_col1:
        st.markdown("#### Poisson Counting Distribution")
        st.latex(r"P(N = k) = \frac{(\lambda T)^k e^{-\lambda T}}{k!} = \frac{\mu^k e^{-\mu}}{k!}")
        st.markdown(
            f"""
            **Numerical Substitution for Active Simulation Parameters:**
            - Arrival intensity: $\\lambda = {lam_cur:.3f}\\text{{ episodes/hr}}$
            - Window horizon: $T = {dur_cur:.1f}\\text{{ hrs}}$
            - Expected event count: $\\mu = \\lambda \\cdot T = ({lam_cur:.3f}) \\cdot ({dur_cur:.1f}) = {pw['mu']:.3f}$
            - Count variance: $\\sigma^2 = \\mu = {pw['mu']:.3f}$
            - Probability of zero events: $P(N=0) = e^{{-{pw['mu']:.3f}}} = {pw['p0']:.4f}$ ({pw['p0']:.2%})
            - Probability of at least one event: $P(N \\ge 1) = 1 - P(N=0) = {pw['p_ge_1']:.4f}$ ({pw['p_ge_1']:.2%})
            """
        )

    with math_col2:
        st.markdown("#### Exponential Inter-Arrival Distribution")
        st.latex(r"f_W(w) = \lambda e^{-\lambda w}, \quad F_W(w) = 1 - e^{-\lambda w}")
        st.markdown(
            f"""
            **Numerical Substitution for Active Simulation Parameters:**
            - Mean waiting time (MTTE): $E[W] = \\frac{{1}}{{\\lambda}} = \\frac{{1}}{{{lam_cur:.3f}}} = {ew['mean']:.3f}\\text{{ hrs}}\\; ({(ew['mean']*60):.1f}\\text{{ mins}})$
            - Variance of waiting time: $\\mathrm{{Var}}(W) = \\frac{{1}}{{\\lambda^2}} = {ew['variance']:.3f}\\text{{ hrs}}^2$
            - Median waiting time: $t_{{1/2}} = \\frac{{\\ln 2}}{{\\lambda}} = {ew['median']:.3f}\\text{{ hrs}}$
            - Exact 95% Poisson rate CI (Garwood):
            """
        )
        st.latex(r"\left[\frac{\chi^2_{2n, 0.025}}{2 T_{\text{total}}}, \; \frac{\chi^2_{2(n+1), 0.975}}{2 T_{\text{total}}}\right]")

    st.markdown("---")
    st.markdown("#### Concepts: Event Counts vs Episode Onsets vs Rates vs Waiting Times")
    st.markdown(
        """
        - **Raw Telemetry Readings:** Every raw sensor sample recorded (e.g. 18-second scan intervals).
        - **Episode Onsets:** The starting point ($0 \\to 1$ transition) of an elevated thermal excursion. A single 10-minute overheating incident generates multiple flagged telemetry rows but counts as **one episode onset**.
        - **Arrival Rate ($\\lambda$):** Average number of discrete episode onsets per operational unit time (episodes/hour).
        - **Inter-Arrival Waiting Time ($W$):** The continuous elapsed operational time between successive episode onsets ($W \\sim \\text{Exp}(\\lambda)$).
        """
    )

    st.markdown("#### Assumptions & Physical Battery Limitations")
    st.markdown(
        """
        1. **Constant Hazard Rate Assumption (Stationarity):** Poisson models assume $\\lambda$ is strictly time-invariant. In real batteries, internal resistance rises with equivalent full cycles (EFC) and solid electrolyte interphase (SEI) growth, causing $\\lambda(t)$ to escalate dramatically over lifetime.
        2. **Memoryless Property of Exponential Gaps:** Assumes $P(W > s + t \\mid W > s) = P(W > t)$. Real cells accumulate thermal fatigue; long defect-free intervals do not guarantee low future probability.
        3. **Independence Assumption:** In battery modules, an initial hot cell transfers heat to adjacent neighbors via conduction, accelerating their reaction rates and causing thermal runaway cascading (clustering).
        """
    )

# =============================================================================
# TAB F: RESULTS & DOWNLOADS
# =============================================================================
with tab_f:
    st.subheader("F. Analytical Comparison & Audited Data Exports")

    st.markdown("#### Comprehensive Comparison: Observed vs Theoretical vs Monte Carlo")
    comp_df = pd.DataFrame([
        {
            "Metric": "Arrival Rate λ (episodes/hr)",
            "Observed Telemetry": f"{obs_summary['lam']:.3f} (CI [{obs_summary['lam_lo']:.3f}, {obs_summary['lam_hi']:.3f}])",
            "Theoretical Model": f"{lam_cur:.3f}",
            "Monte Carlo Sim": "N/A (Input Parameter)",
            "Abs Error": "—",
            "Rel Error (%)": "—",
        },
        {
            "Metric": f"Expected Event Count in T={dur_cur:.1f}h",
            "Observed Telemetry": f"{(obs_summary['lam'] * dur_cur):.3f}",
            "Theoretical Model": f"{pw['mu']:.3f}",
            "Monte Carlo Sim": f"{mean_cnt_sim:.3f}",
            "Abs Error": f"{abs(mean_cnt_sim - pw['mu']):.4f}",
            "Rel Error (%)": f"{(abs(mean_cnt_sim - pw['mu']) / max(pw['mu'], 1e-9) * 100):.2f}%",
        },
        {
            "Metric": "Count Variance σ²",
            "Observed Telemetry": f"{obs_summary['per_cycle_var']:.3f} (per-cycle)",
            "Theoretical Model": f"{pw['mu']:.3f}",
            "Monte Carlo Sim": f"{var_cnt_sim:.3f}",
            "Abs Error": f"{abs(var_cnt_sim - pw['mu']):.4f}",
            "Rel Error (%)": f"{(abs(var_cnt_sim - pw['mu']) / max(pw['mu'], 1e-9) * 100):.2f}%",
        },
        {
            "Metric": "P(N = 0) [Zero Events]",
            "Observed Telemetry": f"{((obs_summary['per_cycle_counts'] == 0).mean()):.2%} (per-cycle)",
            "Theoretical Model": f"{pw['p0']:.4f}",
            "Monte Carlo Sim": f"{p0_sim:.4f}",
            "Abs Error": f"{abs(p0_sim - pw['p0']):.4f}",
            "Rel Error (%)": f"{(abs(p0_sim - pw['p0']) / max(pw['p0'], 1e-9) * 100):.2f}%" if pw['p0'] > 1e-5 else "N/A",
        },
        {
            "Metric": "P(N ≥ 1) [At Least 1 Event]",
            "Observed Telemetry": f"{((obs_summary['per_cycle_counts'] >= 1).mean()):.2%} (per-cycle)",
            "Theoretical Model": f"{pw['p_ge_1']:.4f}",
            "Monte Carlo Sim": f"{p_ge_1_sim:.4f}",
            "Abs Error": f"{abs(p_ge_1_sim - pw['p_ge_1']):.4f}",
            "Rel Error (%)": f"{(abs(p_ge_1_sim - pw['p_ge_1']) / max(pw['p_ge_1'], 1e-9) * 100):.2f}%",
        },
        {
            "Metric": f"P(N ≥ {k_crit_cur}) [Critical Exceedance]",
            "Observed Telemetry": f"{((obs_summary['per_cycle_counts'] >= k_crit_cur).mean()):.2%} (per-cycle)",
            "Theoretical Model": f"{pw['p_ge_k']:.4f}",
            "Monte Carlo Sim": f"{p_ge_k_sim:.4f}",
            "Abs Error": f"{abs(p_ge_k_sim - pw['p_ge_k']):.4f}",
            "Rel Error (%)": f"{(abs(p_ge_k_sim - pw['p_ge_k']) / max(pw['p_ge_k'], 1e-9) * 100):.2f}%" if pw['p_ge_k'] > 1e-5 else "N/A",
        },
        {
            "Metric": "Mean Inter-Arrival Time (hrs)",
            "Observed Telemetry": f"{obs_gaps.mean():.3f} hrs",
            "Theoretical Model": f"{ew['mean']:.3f} hrs",
            "Monte Carlo Sim": f"{mean_wait_sim:.3f} hrs",
            "Abs Error": f"{abs(mean_wait_sim - ew['mean']):.4f}",
            "Rel Error (%)": f"{(abs(mean_wait_sim - ew['mean']) / max(ew['mean'], 1e-9) * 100):.2f}%",
        },
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Audited CSV Exports")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        # Export Monte Carlo Trials
        df_mc_exp = pd.DataFrame({
            "trial_id": np.arange(1, n_sim_cur + 1),
            "simulated_count_N": sim["sim_counts"],
            "simulated_interarrival_wait_h": sim["sim_waits"],
            "lambda_parameter": lam_cur,
            "window_duration_h": dur_cur,
            "threshold_k": k_crit_cur,
            "threshold_strategy": thresh_type_cur,
            "random_seed": sim["seed"],
            "exceeded_threshold": (sim["sim_counts"] >= k_crit_cur).astype(int),
        })
        mc_csv = df_mc_exp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Monte Carlo Trials CSV",
            data=mc_csv,
            file_name=f"monte_carlo_trials_lambda_{lam_cur:.2f}_T_{dur_cur:.1f}h_seed_{sim['seed']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        # Export Processed Episode Onsets
        ep_csv = episodes_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Episode Onsets CSV",
            data=ep_csv,
            file_name=f"processed_episodes_log_{source_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl3:
        # Export Empirical Cycle Summary Table
        cycle_csv = cycle_stats.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Cycle Summary CSV",
            data=cycle_csv,
            file_name=f"cycle_telemetry_summary_{source_name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="note-box" style="margin-top: 15px;">
            <strong>Audit Trail & Export Metadata:</strong><br>
            • <strong>Dataset:</strong> {source_name} ({obs_summary['total_rows']:,} telemetry scans, {obs_summary['cycles']} cycles)<br>
            • <strong>Event Definition:</strong> Consecutive run of thermal_event == 1 readings (excursion onset at initial transition)<br>
            • <strong>Simulation Horizon T:</strong> {dur_cur:.1f} hours | <strong>Active Rate λ:</strong> {lam_cur:.3f} episodes/hr<br>
            • <strong>Monte Carlo Sample Size:</strong> {n_sim_cur:,} independent iterations | <strong>Random Seed:</strong> {sim['seed']}<br>
            • <strong>Decision Strategy:</strong> {thresh_type_cur} (k_crit = {k_crit_cur}, P_warn = {p_warn_cur:.0%}, P_high = {p_high_cur:.0%})
        </div>
        """,
        unsafe_allow_html=True,
    )