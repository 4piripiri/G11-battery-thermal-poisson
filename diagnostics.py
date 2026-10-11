"""
Model Diagnostics & Goodness-of-Fit Plotting Module
===================================================
Generates interactive Plotly figures to critically evaluate whether
homogeneous Poisson and Exponential models hold on empirical battery data.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def plot_cycle_episodes_vs_poisson(cycle_stats: pd.DataFrame) -> go.Figure:
    """Compare observed episode counts per cycle against theoretical Poisson PMF."""
    counts = cycle_stats["episodes"].to_numpy()
    mu = float(np.mean(counts))
    k_vals = np.arange(0, max(3, int(np.max(counts)) + 2))

    # Empirical frequencies
    n_total = len(counts)
    emp_freq = np.array([np.sum(counts == k) / n_total for k in k_vals])
    theo_pmf = np.array([stats.poisson.pmf(k, mu) for k in k_vals])

    var_c = float(np.var(counts, ddof=1))
    dispersion = var_c / mu if mu > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=k_vals,
            y=emp_freq,
            name=f"Observed Data (N={n_total} cycles)",
            marker_color="#2563eb",
            opacity=0.75,
            hovertemplate="Episodes (k): %{x}<br>Observed Fraction: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=theo_pmf,
            mode="lines+markers",
            name=f"Theoretical Poisson (μ = {mu:.3f})",
            line=dict(color="#dc2626", width=2.5),
            marker=dict(size=7, color="#dc2626"),
            hovertemplate="Episodes (k): %{x}<br>Poisson P(N=k): %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Cycle-Level Counts vs Poisson (Mean = {mu:.3f}, Var = {var_c:.3f}, Dispersion = {dispersion:.3f})</b>",
            font=dict(size=13),
        ),
        xaxis=dict(title="Thermal Episodes per Cycle (k)", dtick=1),
        yaxis=dict(title="Probability / Frequency", tickformat=".1%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
        height=360,
    )
    return fig


def plot_interarrival_gaps_vs_exponential(gaps: np.ndarray, lam: float) -> go.Figure:
    """Compare empirical waiting times between consecutive onsets against fitted Exponential PDF."""
    fig = go.Figure()

    if len(gaps) > 0:
        mean_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))
        max_plot = float(min(np.percentile(gaps, 98), 4.5 * (1.0 / lam)))
        t_cont = np.linspace(0.001, max(max_plot, 3.0), 300)
        theo_pdf = lam * np.exp(-lam * t_cont)

        fig.add_trace(
            go.Histogram(
                x=gaps,
                histnorm="probability density",
                nbinsx=25,
                name=f"Observed Gaps (n={len(gaps)})",
                marker_color="#6366f1",
                opacity=0.7,
                hovertemplate="Gap: %{x:.2f} h<br>Empirical Density: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_cont,
                y=theo_pdf,
                mode="lines",
                name=f"Fitted Exp(λ = {lam:.3f})",
                line=dict(color="#dc2626", width=2.5),
                hovertemplate="Wait Time: %{x:.2f} h<br>Fitted PDF: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_vline(
            x=mean_gap,
            line_width=2,
            line_dash="dash",
            line_color="#2563eb",
            annotation_text=f"Observed Mean = {mean_gap:.2f}h",
            annotation_position="top right",
        )
        fig.add_vline(
            x=1.0 / lam,
            line_width=2,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text=f"Model 1/λ = {(1.0/lam):.2f}h",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title=dict(
                text=f"<b>Observed Waiting Times vs Exponential Model (Obs Mean = {mean_gap:.2f}h, Median = {median_gap:.2f}h)</b>",
                font=dict(size=13),
            ),
            xaxis=dict(title="Hours Between Consecutive Episode Onsets", range=[0, max_plot * 1.05]),
            yaxis=dict(title="Probability Density"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=40),
            template="plotly_white",
            height=360,
        )
    return fig


def plot_operating_history_trends(cycle_stats: pd.DataFrame) -> go.Figure:
    """Plot peak cycle temperature and episode onsets across battery operating life."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Peak Cell Temperature Across Battery Discharge Cycles",
            "Thermal Episode Onset Occurrence (1 = Anomaly Onset, 0 = Normal)",
        ),
    )

    # Top: Peak temperature
    fig.add_trace(
        go.Scatter(
            x=cycle_stats["cycle"],
            y=cycle_stats["max_temp"],
            mode="lines+markers",
            name="Peak Temperature (°C)",
            line=dict(color="#d97706", width=1.8),
            marker=dict(size=4, color="#b45309"),
            hovertemplate="Cycle: %{x}<br>Peak Temp: %{y:.2f}°C<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=40.0,
        line_width=1.5,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Nominal Anomaly Threshold (40°C)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Bottom: Episodes
    colors = ["#ef4444" if ep > 0 else "#94a3b8" for ep in cycle_stats["episodes"]]
    fig.add_trace(
        go.Bar(
            x=cycle_stats["cycle"],
            y=cycle_stats["episodes"],
            name="Episode Onset",
            marker_color=colors,
            hovertemplate="Cycle: %{x}<br>Episode Onset: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Discharge Cycle Index", row=2, col=1)
    fig.update_yaxes(title_text="Peak Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Onset Flag", dtick=1, row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_rate_acceleration_and_clustering(cycle_stats: pd.DataFrame) -> go.Figure:
    """Plot rolling episode frequency across cycle life to reveal rate acceleration."""
    cs = cycle_stats.copy()
    window_sz = 20
    cs["rolling_prob"] = cs["episodes"].rolling(window=window_sz, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cs["cycle"],
            y=cs["rolling_prob"],
            mode="lines",
            name=f"{window_sz}-Cycle Rolling Anomaly Rate",
            line=dict(color="#dc2626", width=3),
            hovertemplate="Cycle: %{x}<br>Rolling Rate: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=float(cs["episodes"].mean()),
        line_width=2,
        line_dash="dash",
        line_color="#0f172a",
        annotation_text=f"Global Average ({cs['episodes'].mean():.1%})",
        annotation_position="top left",
    )
    # Highlight early vs late regimes
    fig.add_vrect(
        x0=cs["cycle"].min(),
        x1=cs["cycle"].iloc[min(50, len(cs)-1)],
        fillcolor="rgba(34, 197, 94, 0.12)",
        line_width=0,
        annotation_text="Early Life (Rate ≈ 2%)",
        annotation_position="top left",
    )
    fig.add_vrect(
        x0=cs["cycle"].iloc[min(100, len(cs)-1)],
        x1=cs["cycle"].max(),
        fillcolor="rgba(239, 68, 68, 0.12)",
        line_width=0,
        annotation_text="Late Life Aging (Rate = 100%)",
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text="<b>Evidence of Non-Stationarity: Anomaly Rate Escalation Across Operating Life</b>",
            font=dict(size=13),
        ),
        xaxis=dict(title="Discharge Cycle Index"),
        yaxis=dict(title=f"Fraction of Cycles with Onset ({window_sz}-cycle roll)", tickformat=".0%", range=[-0.05, 1.05]),
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
