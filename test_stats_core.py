"""Checks the code against the hand-calculated worked example (lambda = 1.8)."""
import math
from stats_core import (poisson_pmf, poisson_sf_ge, exp_cdf, exp_survival,
                            exp_mean, estimate_lambda, observed_vs_expected,
                            chi_square_gof, thermal_risk_decision)


def test_poisson_hand_values():
    assert round(poisson_pmf(0, 1.8), 4) == 0.1653
    assert round(poisson_pmf(1, 1.8), 4) == 0.2975
    assert round(poisson_pmf(2, 1.8), 4) == 0.2678
    assert round(poisson_sf_ge(3, 1.8), 4) == 0.2694


def test_exponential_hand_values():
    assert round(exp_mean(1.8), 3) == 0.556
    assert round(exp_cdf(0.5, 1.8), 4) == 0.5934
    assert round(exp_survival(1.0, 1.8), 4) == 0.1653


def test_pmf_sums_to_one():
    assert abs(sum(poisson_pmf(k, 2.3) for k in range(60)) - 1) < 1e-12


def test_lambda_is_mean():
    assert estimate_lambda([0, 2, 4]) == 2


def test_decision_rule():
    status, p = thermal_risk_decision(1.8, 3, 0.25)
    assert status == "HIGH RISK" and round(p, 4) == 0.2694
    assert thermal_risk_decision(0.5, 3, 0.25)[0] == "NORMAL"


def test_gof_table_totals():
    counts = [0, 1, 2, 3, 1, 0, 2, 4, 1, 2] * 13
    lam = estimate_lambda(counts)
    rows = observed_vs_expected(counts, lam)
    assert sum(r["observed"] for r in rows) == len(counts)
    assert abs(sum(r["expected"] for r in rows) - len(counts)) < 1e-9
    stat, dof, p = chi_square_gof(rows)
    assert stat >= 0 and dof >= 1 and 0 <= p <= 1
