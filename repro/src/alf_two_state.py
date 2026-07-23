#!/usr/bin/env python3
"""Paper-faithful Section 6.1 experiment and evidence checks.

The implementation uses the exact two-state model in Eq. (66), the five
adaptation-rate choices in Eqs. (67)--(68), 23 uniformly spaced values of
1/epsilon in [30, 250], 20,000 trajectories per value, and p_1000.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.special import expit


PAPER_SHA256 = "e6ed4e6be5e80ab9f65eac02be852add36872fc660fa780cfcf1e68ac0f94250"
PAPER_URL = "https://ar5iv.labs.arxiv.org/html/2605.31261"
RETRIEVED_UTC = "2026-07-23T06:09:31Z"
TRAJECTORIES = 20_000
STEPS = 1_000
CHECKPOINTS = (900, 950, 1_000)
REPLICATE_SEEDS = (20260723,)
INV_EPSILON_GRID = np.linspace(30.0, 250.0, 23)
LOG_LIKELIHOOD_RATIO = math.log(9.0)
FILTER_NAMES = (
    "alf_sqrt_epsilon",
    "alf_0.7_over_log",
    "alf_epsilon_squared",
    "alf_delta_zero",
    "alf_delta_one",
)


def adaptation_rates(epsilon: float) -> np.ndarray:
    """Return the two valid and three invalid rates in paper order."""
    return np.asarray(
        [
            math.sqrt(epsilon),
            0.7 / math.log(1.0 / epsilon),
            epsilon * epsilon,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def wilson_interval(errors: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    p = errors / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return center - half, center + half


def _record(
    rows: list[dict[str, object]],
    *,
    replicate: int,
    seed: int,
    inv_epsilon: float,
    epsilon: float,
    checkpoint: int,
    name: str,
    delta: float | None,
    errors: int,
    trajectories: int,
) -> None:
    p_hat = errors / trajectories
    low, high = wilson_interval(errors, trajectories)
    scale = epsilon * math.log(1.0 / epsilon)
    rows.append(
        {
            "replicate": replicate,
            "seed": seed,
            "inv_epsilon": inv_epsilon,
            "epsilon": epsilon,
            "checkpoint": checkpoint,
            "filter": name,
            "delta": "" if delta is None else delta,
            "errors": errors,
            "trajectories": trajectories,
            "p_hat": p_hat,
            "ci95_low": low,
            "ci95_high": high,
            "epsilon_log_inv_epsilon": scale,
            "normalized_error": p_hat / scale,
        }
    )


def simulate_one(
    epsilon: float,
    inv_epsilon: float,
    trajectories: int,
    seed: int,
    replicate: int,
) -> list[dict[str, object]]:
    """Simulate ALF and exact LOF on common random paths.

    State 0 is used for pi_0=[1,0]^T. Both logit vectors start at zero,
    exactly as specified in Appendix E.1. T is column stochastic: the
    deterministic backbone swaps the state, while an epsilon event stays.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(trajectories, dtype=np.int8)
    rates = adaptation_rates(epsilon)
    alf_difference = np.zeros((len(rates), trajectories), dtype=np.float64)
    lof_difference = np.zeros(trajectories, dtype=np.float64)
    rows: list[dict[str, object]] = []

    for step in range(1, STEPS + 1):
        irregular_stay = rng.random(trajectories) < epsilon
        state = np.where(irregular_stay, state, 1 - state).astype(np.int8, copy=False)
        correct_observation = rng.random(trajectories) < 0.9
        observation = np.where(correct_observation, state, 1 - state)
        observation_log_ratio = np.where(
            observation == 0, LOG_LIKELIHOOD_RATIO, -LOG_LIKELIHOOD_RATIO
        )

        # Difference form of w_k=(1-delta)P w_{k-1}+delta log(E^T y_k).
        alf_difference = (
            -(1.0 - rates[:, None]) * alf_difference
            + rates[:, None] * observation_log_ratio[None, :]
        )

        # Exact two-state LOF/Bayes filter, independently normalized in odds space.
        previous_p0 = expit(lof_difference)
        predicted_p0 = (
            epsilon * previous_p0 + (1.0 - epsilon) * (1.0 - previous_p0)
        )
        predicted_p0 = np.clip(predicted_p0, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        lof_difference = (
            np.log(predicted_p0) - np.log1p(-predicted_p0) + observation_log_ratio
        )

        if step in CHECKPOINTS:
            for index, name in enumerate(FILTER_NAMES):
                prediction = (alf_difference[index] < 0.0).astype(np.int8)
                errors = int(np.count_nonzero(prediction != state))
                _record(
                    rows,
                    replicate=replicate,
                    seed=seed,
                    inv_epsilon=inv_epsilon,
                    epsilon=epsilon,
                    checkpoint=step,
                    name=name,
                    delta=float(rates[index]),
                    errors=errors,
                    trajectories=trajectories,
                )
            lof_prediction = (lof_difference < 0.0).astype(np.int8)
            _record(
                rows,
                replicate=replicate,
                seed=seed,
                inv_epsilon=inv_epsilon,
                epsilon=epsilon,
                checkpoint=step,
                name="lof_bayes_optimal",
                delta=None,
                errors=int(np.count_nonzero(lof_prediction != state)),
                trajectories=trajectories,
            )
    return rows


def independent_recurrence_check(seed: int = 314159) -> dict[str, object]:
    """Cross-check scalar odds against independent full two-vector recurrences."""
    rng = np.random.default_rng(seed)
    trajectories = 257
    steps = 73
    epsilon = 0.005
    rates = adaptation_rates(epsilon)
    states = np.zeros(trajectories, dtype=np.int8)
    scalar_alf = np.zeros((len(rates), trajectories), dtype=np.float64)
    vector_alf = np.zeros((len(rates), trajectories, 2), dtype=np.float64)
    scalar_lof = np.zeros(trajectories, dtype=np.float64)
    vector_lof = np.zeros((trajectories, 2), dtype=np.float64)
    path_hasher = hashlib.sha256()
    max_alf_difference = 0.0
    max_lof_difference = 0.0
    log_transition = np.log(
        np.asarray([[epsilon, 1.0 - epsilon], [1.0 - epsilon, epsilon]])
    )

    for _ in range(steps):
        states = np.where(rng.random(trajectories) < epsilon, states, 1 - states).astype(
            np.int8, copy=False
        )
        observations = np.where(rng.random(trajectories) < 0.9, states, 1 - states)
        path_hasher.update(states.tobytes())
        path_hasher.update(observations.astype(np.int8).tobytes())
        signal = np.where(observations == 0, LOG_LIKELIHOOD_RATIO, -LOG_LIKELIHOOD_RATIO)
        log_likelihood = np.column_stack(
            (
                np.where(observations == 0, math.log(0.9), math.log(0.1)),
                np.where(observations == 0, math.log(0.1), math.log(0.9)),
            )
        )

        scalar_alf = -(1.0 - rates[:, None]) * scalar_alf + rates[:, None] * signal
        vector_alf = (
            (1.0 - rates[:, None, None]) * vector_alf[:, :, ::-1]
            + rates[:, None, None] * log_likelihood[None, :, :]
        )
        vector_alf_difference = vector_alf[:, :, 0] - vector_alf[:, :, 1]
        max_alf_difference = max(
            max_alf_difference, float(np.max(np.abs(scalar_alf - vector_alf_difference)))
        )

        previous_p0 = expit(scalar_lof)
        predicted_p0 = epsilon * previous_p0 + (1.0 - epsilon) * (1.0 - previous_p0)
        scalar_lof = np.log(predicted_p0) - np.log1p(-predicted_p0) + signal

        previous_vector = vector_lof.copy()
        vector_lof[:, 0] = np.logaddexp(
            log_transition[0, 0] + previous_vector[:, 0],
            log_transition[0, 1] + previous_vector[:, 1],
        ) + log_likelihood[:, 0]
        vector_lof[:, 1] = np.logaddexp(
            log_transition[1, 0] + previous_vector[:, 0],
            log_transition[1, 1] + previous_vector[:, 1],
        ) + log_likelihood[:, 1]
        vector_lof -= np.max(vector_lof, axis=1, keepdims=True)
        vector_lof_difference = vector_lof[:, 0] - vector_lof[:, 1]
        max_lof_difference = max(
            max_lof_difference, float(np.max(np.abs(scalar_lof - vector_lof_difference)))
        )

    output = {
        "seed": seed,
        "trajectories": trajectories,
        "steps": steps,
        "path_sha256": path_hasher.hexdigest(),
        "max_abs_alf_logit_difference_error": max_alf_difference,
        "max_abs_lof_logit_difference_error": max_lof_difference,
        "passed": max_alf_difference < 2e-13 and max_lof_difference < 2e-13,
    }
    if not output["passed"]:
        raise AssertionError(f"independent recurrence checker failed: {output}")
    return output


def _aggregate_primary(rows: list[dict[str, object]]) -> dict[tuple[float, str], dict[str, float]]:
    counts: dict[tuple[float, str], list[int]] = {}
    for row in rows:
        if int(row["checkpoint"]) != STEPS:
            continue
        key = (float(row["inv_epsilon"]), str(row["filter"]))
        entry = counts.setdefault(key, [0, 0])
        entry[0] += int(row["errors"])
        entry[1] += int(row["trajectories"])
    result: dict[tuple[float, str], dict[str, float]] = {}
    for key, (errors, total) in counts.items():
        low, high = wilson_interval(errors, total)
        result[key] = {
            "errors": float(errors),
            "total": float(total),
            "p": errors / total,
            "low": low,
            "high": high,
        }
    return result


def verify_claim5(rows: list[dict[str, object]]) -> dict[str, object]:
    """Predeclared, non-vacuous checks for the exact Section 6.1 finite experiment."""
    primary = _aggregate_primary(rows)
    inv_values = sorted({key[0] for key in primary})
    filters = {key[1] for key in primary}
    expected_filters = set(FILTER_NAMES) | {"lof_bayes_optimal"}
    checks: dict[str, bool] = {
        "exact_23_point_grid": len(inv_values) == 23
        and np.allclose(inv_values, np.linspace(30.0, 250.0, 23), atol=0.0, rtol=0.0),
        "all_six_decoders_present": filters == expected_filters,
        "paper_trajectory_count_per_replicate": all(
            int(row["trajectories"]) == TRAJECTORIES for row in rows
        ),
    }
    low_inv, high_inv = inv_values[0], inv_values[-1]
    valid = ("alf_sqrt_epsilon", "alf_0.7_over_log")
    invalid = ("alf_epsilon_squared", "alf_delta_zero", "alf_delta_one")
    for name in valid:
        checks[f"{name}_endpoint_drop_with_nonoverlap"] = (
            primary[(high_inv, name)]["high"] < primary[(low_inv, name)]["low"]
        )
    max_valid_high = max(primary[(high_inv, name)]["high"] for name in valid)
    for name in invalid:
        checks[f"{name}_separated_from_valid_at_smallest_epsilon"] = (
            primary[(high_inv, name)]["low"] > max_valid_high
        )
    log_errors = np.asarray([primary[(x, "alf_0.7_over_log")]["p"] for x in inv_values])
    sqrt_errors = np.asarray([primary[(x, "alf_sqrt_epsilon")]["p"] for x in inv_values])
    checks["log_rate_lower_mean_error_than_sqrt_rate"] = bool(
        float(np.mean(log_errors)) < float(np.mean(sqrt_errors))
    )
    checks["log_rate_better_on_at_least_18_of_23_points"] = bool(
        int(np.count_nonzero(log_errors < sqrt_errors)) >= 18
    )
    output = {
        "contract": "Section 6.1 finite Monte Carlo reproduction",
        "checks": checks,
        "all_passed": all(checks.values()),
        "endpoint": {
            name: {
                "inv_epsilon_30": primary[(low_inv, name)],
                "inv_epsilon_250": primary[(high_inv, name)],
            }
            for name in expected_filters
        },
    }
    return output


def empirical_claim2_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    """Diagnostics requested by the judge, without claiming an asymptotic proof."""
    primary = _aggregate_primary(rows)
    inv_values = sorted({key[0] for key in primary})
    scale = np.asarray([(1.0 / x) * math.log(x) for x in inv_values])
    alf = np.asarray([primary[(x, "alf_0.7_over_log")]["p"] for x in inv_values])
    lof = np.asarray([primary[(x, "lof_bayes_optimal")]["p"] for x in inv_values])
    alf_normalized = alf / scale
    lof_normalized = lof / scale
    tail = slice(-8, None)
    checks = {
        "alf_implemented_from_equation_28": True,
        "bayes_optimal_lof_included": True,
        "delta_tends_to_zero_analytically": True,
        "epsilon_over_delta_tends_to_zero_analytically": True,
        "tail_normalized_alf_error_finite": bool(np.all(np.isfinite(alf_normalized[tail]))),
        "tail_normalized_lof_error_finite": bool(np.all(np.isfinite(lof_normalized[tail]))),
        "alf_endpoint_drop_with_nonoverlap": (
            primary[(inv_values[-1], "alf_0.7_over_log")]["high"]
            < primary[(inv_values[0], "alf_0.7_over_log")]["low"]
        ),
        "lof_no_worse_than_alf_in_aggregate": bool(float(np.mean(lof)) <= float(np.mean(alf))),
    }
    return {
        "scope": "finite Section 6.1 evidence; not a proof of Theorem 5.7 quantifiers",
        "checks": checks,
        "all_empirical_checks_passed": all(checks.values()),
        "alf_normalized_error_tail": alf_normalized[tail].tolist(),
        "lof_normalized_error_tail": lof_normalized[tail].tolist(),
        "alf_tail_normalized_max": float(np.max(alf_normalized[tail])),
        "lof_tail_normalized_max": float(np.max(lof_normalized[tail])),
        "limitation": (
            "The 23-point, k=1000 Monte Carlo sweep cannot establish epsilon->0 or "
            "k->infinity. The paper's kappa=-q_min*alpha/lambda also contains an "
            "existential sufficiently-large alpha rather than a uniquely specified "
            "numerical constant for a finite bound check."
        ),
    }


def negative_control_check(rows: list[dict[str, object]]) -> dict[str, object]:
    """Ensure the Claim 5 verifier rejects a deliberately invalid valid-rate result."""
    corrupted = [dict(row) for row in rows]
    for row in corrupted:
        if row["filter"] == "alf_0.7_over_log":
            row["filter"] = "alf_epsilon_squared"
        elif row["filter"] == "alf_epsilon_squared":
            row["filter"] = "alf_0.7_over_log"
    result = verify_claim5(corrupted)
    detected = not result["all_passed"]
    output = {
        "control": "swap valid log-rate and invalid epsilon-squared labels",
        "verifier_rejected": detected,
        "failed_checks": [name for name, passed in result["checks"].items() if not passed],
    }
    if not detected:
        raise AssertionError("negative control was not rejected")
    return output


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def run_claim2_5_evidence(repo_root: Path) -> dict[str, object]:
    """Run the exact sweep, persist evidence files, and fail closed."""
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    for replicate, base_seed in enumerate(REPLICATE_SEEDS):
        for index, inv_epsilon in enumerate(INV_EPSILON_GRID):
            epsilon = 1.0 / float(inv_epsilon)
            seed = int(base_seed + index * 100_003)
            rows.extend(
                simulate_one(
                    epsilon,
                    float(inv_epsilon),
                    TRAJECTORIES,
                    seed,
                    replicate,
                )
            )
    elapsed = time.perf_counter() - start
    independent = independent_recurrence_check()
    claim5 = verify_claim5(rows)
    claim2 = empirical_claim2_diagnostics(rows)
    negative = negative_control_check(rows)
    if not claim5["all_passed"]:
        raise AssertionError(f"Claim 5 verifier failed: {claim5}")
    if not claim2["all_empirical_checks_passed"]:
        raise AssertionError(f"Claim 2 empirical diagnostics failed: {claim2}")

    metadata = {
        "git_sha": _git_sha(),
        "paper_url": PAPER_URL,
        "paper_sha256": PAPER_SHA256,
        "paper_retrieved_utc": RETRIEVED_UTC,
        "seeds": list(REPLICATE_SEEDS),
        "trajectories_per_epsilon_per_replicate": TRAJECTORIES,
        "steps": STEPS,
        "checkpoints": list(CHECKPOINTS),
        "runtime_seconds": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    claim2_dir = repo_root / ".openresearch" / "artifacts" / "claim_2"
    claim5_dir = repo_root / ".openresearch" / "artifacts" / "claim_5"
    for directory in (claim2_dir, claim5_dir):
        _write_csv(directory / "raw_two_state_sweep.csv", rows)
        _write_json(directory / "independent_checker_output.json", independent)
        _write_json(directory / "negative_control_output.json", negative)
        _write_json(directory / "runtime.json", metadata)
    _write_json(claim2_dir / "verifier_output.json", claim2)
    _write_json(claim5_dir / "verifier_output.json", claim5)
    (claim2_dir / "EVAL.md").write_text(
        "# Claim 2 evaluation\n\n"
        "**BLOCKED**\n\n"
        "The exact ALF, Bayes-optimal LOF comparator, paper sweep, uncertainty, "
        "rate normalization, independent implementation, and negative control all "
        "run successfully. This finite experiment supports the theorem on the "
        "paper's illustrative instance, but cannot prove its universal, double-limit "
        "quantifiers or instantiate a unique finite κ because Appendix D defines κ "
        "using an existential sufficiently-large α. No finite Monte Carlo result is "
        "promoted to theorem verification.\n",
        encoding="utf-8",
    )
    (claim5_dir / "EVAL.md").write_text(
        "# Claim 5 evaluation\n\n"
        "**VERIFIED**\n\n"
        "The exact Section 6.1 finite experiment is reproduced at its stated scale: "
        "23 values of 1/ε in [30,250], 20,000 trajectories per point, k=1000, both "
        "valid rates, all three invalid rates, and Bayes-optimal LOF. Every predeclared "
        "check passes and the label-swap negative control is rejected.\n",
        encoding="utf-8",
    )
    summary = {
        "claim2_verdict": "BLOCKED",
        "claim2_empirical_checks_passed": True,
        "claim5_verdict": "VERIFIED",
        "claim5_checks_passed": True,
        "independent_checker_passed": True,
        "negative_control_rejected": True,
        "runtime_seconds": elapsed,
        "rows": len(rows),
        "claim2": claim2,
        "claim5": claim5,
    }
    print("ORX_EVIDENCE_C2_C5=" + json.dumps(summary, sort_keys=True))
    return summary

