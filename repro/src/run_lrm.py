#!/usr/bin/env python3
"""Verify the two claims of "Why Linear Recurrent Memory Works in PORL" (arXiv 2605.31262).

C1 (exact identity): a LINEAR filter / linear RNN  l_t = T l_{t-1} + log e_t
exactly reproduces the pre-softmax logits (log alpha_t) of the HMM belief vector
when the transition matrix T is DETERMINISTIC. (Reason: the forward recursion's
log-sum-exp collapses to a single term, linearizing in log-space.)

C2 (vanishing error): under a NEARLY-deterministic transition matrix, the linear
filter's state-decoding error vanishes as the matrix -> deterministic.

We verify by computing the belief logits TWO independent ways and comparing:
  - general log-forward (scipy logsumexp; correct for ANY T)  -- the "truth"
  - linear recurrence l_t = T l_{t-1} + log e_t                  -- the "linear filter"
and a negative control: for a STOCHASTIC (non-deterministic) T they must MISMATCH.
"""
import os, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from alf_two_state import run_claim2_5_evidence
from action_controlled import run_claim3_evidence
from claim4_blocker import run_claim4_evidence
from release_checks import validate_release_candidate
from certified_audit import run_certified_audit


def gen_hmm(n, m, deterministic, eps=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # T[i,j] = P(next=i | curr=j), column-stochastic. Deterministic = permutation.
    perm = rng.permutation(n)
    Tdet = np.zeros((n, n)); Tdet[perm, np.arange(n)] = 1.0
    if deterministic:
        T = Tdet
    else:
        Ts = rng.dirichlet(np.ones(n), size=n).T      # random column-stochastic
        T = (1 - eps) * Tdet + eps * Ts
        T = T / T.sum(0, keepdims=True)
    E = rng.random((n, m)) + 0.05                       # emission likelihoods
    pi = np.ones(n) / n
    return T, Tdet, E, pi


def log_forward(T, logE_obs, logpi):
    """General log-forward (logsumexp) -- correct for any T. Returns log alpha_t list."""
    logT = np.log(T, where=T > 0, out=np.full_like(T, -np.inf))
    L = []
    lprev = logpi.copy()
    for le in logE_obs:                     # le = log e_t (vector over states)
        # log alpha_t(i) = le(i) + logsumexp_j(logT[i,j] + lprev[j])
        lcurr = le + logsumexp(logT + lprev[None, :], axis=1)
        L.append(lcurr); lprev = lcurr
    return L


def linear_filter(T, logE_obs, logpi):
    """Linear recurrence l_t = T l_{t-1} + log e_t  (the paper's linear filter)."""
    L = []
    lprev = logpi.copy()
    for le in logE_obs:
        lcurr = T @ lprev + le
        L.append(lcurr); lprev = lcurr
    return L


def max_abs_diff(A, B):
    return float(max(np.max(np.abs(a - b)) for a, b in zip(A, B)))


def tv_mean(A, B):
    """Mean total-variation distance between softmax(A_t) and softmax(B_t)."""
    tvs = []
    for a, b in zip(A, B):
        pa = np.exp(a - a.max()); pa /= pa.sum()
        pb = np.exp(b - b.max()); pb /= pb.sum()
        tvs.append(0.5 * np.abs(pa - pb).sum())
    return float(np.mean(tvs))


def main():
    n, m, T_len, seed = 8, 12, 40, 0
    rng = np.random.default_rng(seed)

    # ---- C1: deterministic T -> linear filter EXACTLY reproduces belief logits ----
    T, Tdet, E, pi = gen_hmm(n, m, deterministic=True, seed=seed)
    obs = rng.integers(0, m, size=T_len)
    logE_obs = [np.log(E[:, o]) for o in obs]
    logpi = np.log(pi)
    true_det = log_forward(T, logE_obs, logpi)        # general logsumexp forward
    lin_det = linear_filter(T, logE_obs, logpi)       # linear recurrence
    c1_maxdiff = max_abs_diff(true_det, lin_det)
    c1_exact = c1_maxdiff < 1e-9

    # ---- Negative control: STOCHASTIC T -> linear filter must MISMATCH ----
    Ts, _, _, _ = gen_hmm(n, m, deterministic=False, eps=1.0, seed=seed + 1)
    obs2 = rng.integers(0, m, size=T_len)
    logE2 = [np.log(E[:, o]) for o in obs2]
    true_stoch = log_forward(Ts, logE2, logpi)
    lin_stoch = linear_filter(Ts, logE2, logpi)       # (mis)applies the linear filter
    control_mismatch = max_abs_diff(true_stoch, lin_stoch)
    control_ok = control_mismatch > 0.1                # must be clearly non-exact

    # ---- C2: nearly-deterministic T_eps -> decoding error vanishes as eps -> 0 ----
    # Fixed obs sequence across eps so only T varies (isolate the determinism effect).
    eps_grid = [0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001]
    obs_e = rng.integers(0, m, size=T_len)
    logEe = [np.log(E[:, o]) for o in obs_e]
    tv_by_eps = []
    for eps in eps_grid:
        Te, _, _, _ = gen_hmm(n, m, deterministic=False, eps=eps, seed=seed)
        true_e = log_forward(Te, logEe, logpi)
        lin_e = linear_filter(Te, logEe, logpi)       # linear filter (assumes ~deterministic)
        tv_by_eps.append(tv_mean(true_e, lin_e))
    c2_vanishes = tv_by_eps[-1] < 0.05 and tv_by_eps[-1] < tv_by_eps[0]

    res = dict(paper="arXiv 2605.31261", n=n, m=m, T=T_len,
               C1=dict(exact_match=bool(c1_exact), max_abs_diff=c1_maxdiff),
               control=dict(stochastic_mismatch=bool(control_ok), max_abs_diff=control_mismatch),
               C2=dict(eps_grid=eps_grid, tv_decoding_error=tv_by_eps,
                       vanishes=bool(c2_vanishes), final_tv=tv_by_eps[-1]))
    out = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "lrm_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("=" * 64)
    print("Linear Recurrent Memory (arXiv 2605.31261) verification")
    print("=" * 64)
    print(f"C1 (deterministic T): linear filter EXACTLY reproduces belief logits:")
    print(f"    max|log_forward - linear_filter| = {c1_maxdiff:.3e}  -> exact: {c1_exact}")
    print(f"Negative control (stochastic T): linear filter must mismatch:")
    print(f"    max|diff| = {control_mismatch:.3e}  -> mismatch confirmed: {control_ok}")
    print(f"C2 (nearly-deterministic): TV decoding error vs eps:")
    for e, tv in zip(eps_grid, tv_by_eps):
        print(f"    eps={e:<6}  TV={tv:.4f}")
    print(f"    -> error vanishes as eps->0: {c2_vanishes} (final TV={tv_by_eps[-1]:.4f})")
    print("=" * 64)
    print("wrote", out)
    repo_root = Path(__file__).resolve().parents[2]
    print("Running paper-faithful Claims 2/5 evidence suite")
    run_claim2_5_evidence(repo_root)
    print("Running action-controlled Claim 3 evidence suite")
    run_claim3_evidence(repo_root)
    print("Running full-scale RingWorld Claim 4 blocker audit")
    run_claim4_evidence(repo_root)
    print("Validating cumulative logbook release candidate")
    validate_release_candidate(repo_root)
    print("Running certified theorem and source-raster audit")
    run_certified_audit(repo_root)


if __name__ == "__main__":
    main()
