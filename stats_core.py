"""
Core statistics for IA-1 (Topic 11: Poisson and Exponential models).

Everything here uses plain Python `math` on purpose, so each formula can be
read, explained in the viva, and checked against the hand calculations.
scipy is used ONLY for the chi-square p-value at the very end.
"""
import math
from statistics import mean


# ---------- Estimation ----------
def estimate_lambda(counts):
    """MLE of the Poisson rate = sample mean of the counts."""
    return mean(counts)


def sample_variance(values):
    """s^2 = sum (x - xbar)^2 / (n - 1)."""
    n = len(values)
    xbar = sum(values) / n
    return sum((x - xbar) ** 2 for x in values) / (n - 1)


def dispersion_index(counts):
    """Variance / mean. Close to 1 supports a Poisson model."""
    return sample_variance(counts) / estimate_lambda(counts)


# ---------- Poisson ----------
def poisson_pmf(k, lam):
    """P(X = k) = e^-lam * lam^k / k!"""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_cdf(k, lam):
    """P(X <= k)."""
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_sf_ge(k, lam):
    """P(X >= k) = 1 - P(X <= k-1)."""
    return 1.0 - poisson_cdf(k - 1, lam)


# ---------- Exponential ----------
def exp_pdf(t, lam):
    """f(t) = lam * e^(-lam t), t >= 0."""
    return lam * math.exp(-lam * t) if t >= 0 else 0.0


def exp_cdf(t, lam):
    """P(T <= t) = 1 - e^(-lam t)."""
    return 1.0 - math.exp(-lam * t) if t >= 0 else 0.0


def exp_survival(t, lam):
    """P(T > t) = e^(-lam t)."""
    return math.exp(-lam * t) if t >= 0 else 1.0


def exp_mean(lam):
    """E[T] = 1 / lam."""
    return 1.0 / lam


# ---------- Goodness of fit ----------
def observed_vs_expected(counts, lam, min_expected=5.0):
    """
    Build the observed vs expected frequency table for a Poisson(lam) fit.
    Categories k = 0..max(counts) with the last one as the tail (>= max).
    Neighbouring categories are merged until every expected count >= min_expected
    (the usual rule of thumb for the chi-square test).
    Returns a list of dicts: label, observed, expected, chi_term.
    """
    n = len(counts)
    kmax = max(counts)
    bins = []
    for k in range(kmax):
        bins.append({"lo": k, "hi": k, "obs": sum(1 for c in counts if c == k),
                     "exp": n * poisson_pmf(k, lam)})
    bins.append({"lo": kmax, "hi": None,  # tail: k >= kmax
                 "obs": sum(1 for c in counts if c >= kmax),
                 "exp": n * poisson_sf_ge(kmax, lam)})

    # merge small categories
    while len(bins) > 2:
        idx = next((i for i, b in enumerate(bins) if b["exp"] < min_expected), None)
        if idx is None:
            break
        j = idx + 1 if idx < len(bins) - 1 else idx - 1
        a, b = sorted((idx, j))
        merged = {"lo": bins[a]["lo"], "hi": bins[b]["hi"],
                  "obs": bins[a]["obs"] + bins[b]["obs"],
                  "exp": bins[a]["exp"] + bins[b]["exp"]}
        bins[a:b + 1] = [merged]

    rows = []
    for b in bins:
        if b["hi"] is None:
            label = f">= {b['lo']}"
        elif b["lo"] == b["hi"]:
            label = f"{b['lo']}"
        else:
            label = f"{b['lo']}-{b['hi']}"
        rows.append({"label": label, "observed": b["obs"], "expected": b["exp"],
                     "chi_term": (b["obs"] - b["exp"]) ** 2 / b["exp"]})
    return rows


def chi_square_gof(rows, estimated_params=1):
    """chi2 = sum (O-E)^2 / E ; dof = categories - 1 - estimated_params."""
    stat = sum(r["chi_term"] for r in rows)
    dof = len(rows) - 1 - estimated_params
    p = None
    if dof >= 1:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, dof))
    return stat, dof, p


# ---------- Engineering / AI decision ----------
def thermal_risk_decision(lam, k_threshold=3, p_threshold=0.25):
    """
    Probability-based engineering decision rule:
      if P(X >= k_threshold events in a discharge cycle)
      > p_threshold -> HIGH RISK

    Returns (status, probability).
    """
    p = poisson_sf_ge(k_threshold, lam)
    status = "HIGH RISK" if p > p_threshold else "NORMAL"
    return status, p
