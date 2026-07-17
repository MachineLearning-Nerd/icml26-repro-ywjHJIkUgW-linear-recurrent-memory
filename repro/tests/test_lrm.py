"""Unit tests for the linear-recurrent-memory verification.

Run:  .venv/bin/python -m pytest repro/tests/test_lrm.py -q
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from run_lrm import gen_hmm, log_forward, linear_filter, max_abs_diff, tv_mean


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
