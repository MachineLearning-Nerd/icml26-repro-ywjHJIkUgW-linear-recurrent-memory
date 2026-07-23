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
from action_controlled import (
    action_model,
    exact_action_identity,
    independent_action_checker,
    permutation_certificate,
)
from claim4_blocker import evaluate_claim4, reject_proxy_as_full_evidence, resource_contract
from release_checks import validate_release_candidate
from certified_audit import (
    action_filter_certificate,
    bounded_counterexample_search,
    digitize_source_figures,
    fixed_filter_certificate,
    proof_negative_controls,
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


def test_action_permutation_certificate():
    permutations, _, emissions = action_model()
    certificate = permutation_certificate(permutations)
    assert certificate["passed"]
    assert np.all(emissions > 0.0)
    assert len({tuple(emissions[:, index]) for index in range(emissions.shape[1])}) == 8


def test_action_arbitrary_initialization_identity():
    result = exact_action_identity()
    assert result["passed"]
    assert result["maximum_absolute_logit_error"] < 1e-12
    assert result["stochastic_action_negative_control_mismatch"] > 0.1


def test_independent_action_index_checker():
    assert independent_action_checker()["passed"]


def test_claim4_resource_lower_bound():
    resources = resource_contract()
    assert resources["figure3_training_runs"] == 56
    assert resources["figure3_environment_steps"] == 1_680_000_000
    assert resources["figure3_sample_epoch_passes_lower_bound"] == 50_400_000_000
    assert resources["minimum_training_runs"] == 32


def test_claim4_proxy_is_rejected():
    assert reject_proxy_as_full_evidence()["verifier_rejected"]


def test_claim4_dossier_is_blocked_not_pass(tmp_path):
    result = evaluate_claim4(tmp_path)
    assert result["claim4_verdict"] == "BLOCKED"
    assert result["blocker_dossier_valid"]


def test_release_candidate_is_fail_closed():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = validate_release_candidate(repo_root)
    assert result["candidate_logbook_valid"]
    assert result["historical_files_hash_identical"]
    assert result["claim_checks"]["claim_2"]["expected_verdict"] == "BLOCKED"


def test_exact_two_state_chernoff_certificate():
    emissions = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    result = fixed_filter_certificate(swap, emissions, q_rate=1.0)
    assert result["passed"]
    assert abs(result["xi"] - 0.7206014018371382) < 1e-10
    assert 0.0 < result["lambda"] < result["xi"]


def test_action_certificate_is_uniform():
    permutations, _, emissions = action_model()
    result = action_filter_certificate(permutations, emissions, q_rate=0.875)
    assert result["passed"]
    assert result["all_actions_preserve_pair_separation"]


def test_bounded_counterexample_search_and_controls():
    assert bounded_counterexample_search()["passed"]
    assert proof_negative_controls()["passed"]


def test_source_raster_decoder_is_rejected_for_ambiguous_convergence_time():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    result = digitize_source_figures(repo_root, tolerance=12.0)
    assert not result["passed"]
    assert not result["checks"]["alf_reaches_0.4_before_lof"]
    assert result["checks"]["deep_alf_final_mean_above_s5_hidden_12"]
