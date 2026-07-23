"""Unit tests for the linear-recurrent-memory verification.

Run:  .venv/bin/python -m pytest repro/tests/test_lrm.py -q
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from run_lrm import gen_hmm, log_forward, linear_filter, max_abs_diff, tv_mean
from alf_two_state import (
    adaptation_rates,
    independent_recurrence_check,
    verify_claim5,
    wilson_interval,
)


def test_deterministic_exact():
    """C1: under deterministic T, linear filter == log-forward (bit-exact)."""
    T, _, E, pi = gen_hmm(6, 8, deterministic=True, seed=1)
    rng = np.random.default_rng(2)
    obs = rng.integers(0, 8, size=30)
    logE = [np.log(E[:, o]) for o in obs]
    logpi = np.log(pi)
    diff = max_abs_diff(log_forward(T, logE, logpi), linear_filter(T, logE, logpi))
    assert diff < 1e-10, f"deterministic T should be exact, got {diff}"


def test_stochastic_mismatch():
    """Negative control: stochastic T -> linear filter != log-forward."""
    T, _, E, pi = gen_hmm(6, 8, deterministic=False, eps=1.0, seed=3)
    rng = np.random.default_rng(4)
    obs = rng.integers(0, 8, size=30)
    logE = [np.log(E[:, o]) for o in obs]
    logpi = np.log(pi)
    diff = max_abs_diff(log_forward(T, logE, logpi), linear_filter(T, logE, logpi))
    assert diff > 1.0, f"stochastic T should mismatch, got {diff}"


def test_vanishing_error():
    """C2: TV decoding error decreases as transitions -> deterministic."""
    n, m = 6, 8
    _, _, E, pi = gen_hmm(n, m, deterministic=True, seed=5)
    rng = np.random.default_rng(6)
    obs = rng.integers(0, m, size=30)
    logE = [np.log(E[:, o]) for o in obs]
    logpi = np.log(pi)
    tvs = []
    for eps in [0.3, 0.03, 0.003]:
        Te, _, _, _ = gen_hmm(n, m, deterministic=False, eps=eps, seed=5)
        tvs.append(tv_mean(log_forward(Te, logE, logpi), linear_filter(Te, logE, logpi)))
    assert tvs[2] < tvs[0], f"error should shrink: {tvs}"


def test_adaptation_rate_conditions():
    """The two paper-valid choices meet delta->0 and epsilon/delta->0 numerically."""
    eps = np.asarray([1e-4, 1e-8, 1e-12])
    sqrt_delta = np.sqrt(eps)
    log_delta = 0.7 / np.log(1.0 / eps)
    assert np.all(np.diff(sqrt_delta) < 0)
    assert np.all(np.diff(log_delta) < 0)
    assert np.all(np.diff(eps / sqrt_delta) < 0)
    assert np.all(np.diff(eps / log_delta) < 0)
    rates = adaptation_rates(0.005)
    assert np.allclose(
        rates,
        [np.sqrt(0.005), 0.7 / np.log(200.0), 0.005**2, 0.0, 1.0],
    )


def test_wilson_interval_contains_estimate():
    low, high = wilson_interval(23, 200)
    assert low < 23 / 200 < high


def test_independent_full_vector_checker():
    result = independent_recurrence_check(seed=271828)
    assert result["passed"]


def test_claim5_verifier_fails_on_empty_evidence():
    """Verifiers fail closed; skipped or absent raw data cannot become a PASS."""
    try:
        result = verify_claim5([])
    except (IndexError, KeyError):
        return
    assert not result["all_passed"]
