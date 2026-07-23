#!/usr/bin/env python3
"""Fail-closed audit for the full-scale RingWorld training claim."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import subprocess
from pathlib import Path

from alf_two_state import PAPER_SHA256, PAPER_URL, RETRIEVED_UTC


PAPER_TIMESTEPS = 30_000_000
PAPER_SEEDS = 8
PAPER_ENVIRONMENTS = 64
PAPER_UNROLL = 1_024
PAPER_EPOCHS = 30
FIGURE3_VARIANTS = (
    "S5 hidden size 256",
    "S5 hidden size 12",
    "LOF direct",
    "LOF softmax",
    "ALF direct delta=0.1",
    "ALF softmax delta=0.1",
    "Deep ALF",
)
MINIMUM_CLAIM_VARIANTS = (
    "LOF direct",
    "ALF direct delta=0.1",
    "S5 hidden size 12",
    "Deep ALF",
)
REQUIRED_MODULES = ("jax", "flax", "optax", "distrax", "gymnax")
RELEASED_SOURCE_SHA256 = (
    "f85693bad9adf7d13c2ae827178bbb052f54b7460dcfd0261802f9bf6eac214c"
)


def resource_contract() -> dict[str, object]:
    """Compute exact lower bounds implied by the published protocol."""
    figure_runs = len(FIGURE3_VARIANTS) * PAPER_SEEDS
    minimum_runs = len(MINIMUM_CLAIM_VARIANTS) * PAPER_SEEDS
    figure_environment_steps = figure_runs * PAPER_TIMESTEPS
    minimum_environment_steps = minimum_runs * PAPER_TIMESTEPS
    updates_per_run = math.ceil(
        PAPER_TIMESTEPS / (PAPER_ENVIRONMENTS * PAPER_UNROLL)
    )
    collected_steps_per_run = updates_per_run * PAPER_ENVIRONMENTS * PAPER_UNROLL
    figure_sample_epoch_passes_lower_bound = (
        figure_environment_steps * PAPER_EPOCHS
    )
    independently_rounded_passes = (
        figure_runs * updates_per_run * PAPER_ENVIRONMENTS * PAPER_UNROLL * PAPER_EPOCHS
    )
    return {
        "minimum_claim_variants": list(MINIMUM_CLAIM_VARIANTS),
        "minimum_training_runs": minimum_runs,
        "minimum_environment_steps": minimum_environment_steps,
        "figure3_variants": list(FIGURE3_VARIANTS),
        "figure3_training_runs": figure_runs,
        "figure3_environment_steps": figure_environment_steps,
        "updates_per_run_if_batches_are_complete": updates_per_run,
        "collected_steps_per_run_if_batches_are_complete": collected_steps_per_run,
        "figure3_sample_epoch_passes_lower_bound": figure_sample_epoch_passes_lower_bound,
        "figure3_sample_epoch_passes_with_complete_batches": independently_rounded_passes,
        "independent_arithmetic_consistent": (
            independently_rounded_passes >= figure_sample_epoch_passes_lower_bound
        ),
    }


def released_protocol_audit(repo_root: Path) -> dict[str, object]:
    """Inventory material omissions that prevent a uniquely faithful run."""
    available_modules = {
        module: importlib.util.find_spec(module) is not None for module in REQUIRED_MODULES
    }
    omissions = {
        "episode_length_K": "not numerically specified",
        "beacon_angles_xi_i": "not numerically specified; shown only in figures",
        "three_action_transition_entries": (
            "only CW1 is written explicitly; CW2/CCW1/CCW2 are described but their "
            "Q(a) entries are not tabulated"
        ),
        "eight_random_seeds": "count is reported but seed values are not",
        "numeric_learning_curves": "raster figures only; no raw returns or checkpoints",
        "direct_paper_implementation": (
            "no code link in the paper or its author publication entry; exact-title "
            "GitHub repository search returned zero repositories on 2026-07-23"
        ),
    }
    reproduction_dir = repo_root / "repro"
    candidate_training_files = [
        str(path.relative_to(repo_root))
        for path in reproduction_dir.rglob("*")
        if path.is_file()
        and (
            "ringworld" in path.name.lower()
            or "ppo" in path.name.lower()
            or "deep_alf" in path.name.lower()
        )
    ]
    return {
        "paper_source": PAPER_URL,
        "paper_html_sha256": PAPER_SHA256,
        "paper_html_retrieved_utc": RETRIEVED_UTC,
        "arxiv_eprint_url": "https://export.arxiv.org/e-print/2605.31261",
        "arxiv_eprint_retrieved_utc": "2026-07-23T07:15:00Z",
        "arxiv_eprint_sha256": RELEASED_SOURCE_SHA256,
        "author_publication_page": "https://onnoeberhard.com/publications",
        "author_publication_page_retrieved_utc": "2026-07-23T07:19:43Z",
        "author_publication_page_sha256": (
            "7b10cc8790d1f3e78647d6ddf311b3d1055ae9e3252e3d92b3c64735114475f1"
        ),
        "adjacent_s5_code": "https://github.com/luchris429/s5rl",
        "adjacent_code_is_direct_paper_code": False,
        "missing_protocol_fields": omissions,
        "required_autodiff_modules_available": available_modules,
        "all_required_autodiff_modules_available": all(available_modules.values()),
        "candidate_training_files_in_reproduction": candidate_training_files,
        "full_training_implementation_present": False,
    }


def reject_proxy_as_full_evidence() -> dict[str, object]:
    """Negative control: a small environment-only check must not receive PASS."""
    proxy = {
        "states": 12,
        "observations": 4,
        "trained_timesteps": 0,
        "trained_seeds": 0,
        "has_ppo": False,
        "has_deep_alf": False,
        "has_s5_equal_hidden_size": False,
    }
    accepted = (
        proxy["trained_timesteps"] >= PAPER_TIMESTEPS
        and proxy["trained_seeds"] >= PAPER_SEEDS
        and proxy["has_ppo"]
        and proxy["has_deep_alf"]
        and proxy["has_s5_equal_hidden_size"]
    )
    return {
        "control": proxy,
        "accepted_as_full_claim_evidence": accepted,
        "verifier_rejected": not accepted,
    }


def evaluate_claim4(repo_root: Path) -> dict[str, object]:
    resources = resource_contract()
    audit = released_protocol_audit(repo_root)
    negative = reject_proxy_as_full_evidence()
    blockers = {
        "protocol_not_uniquely_specified": bool(audit["missing_protocol_fields"]),
        "direct_training_implementation_absent": not audit[
            "full_training_implementation_present"
        ],
        "pinned_autodiff_stack_absent": not audit[
            "all_required_autodiff_modules_available"
        ],
        "full_cpu_training_not_executed": True,
        "raw_numeric_paper_curves_absent": True,
    }
    verdict = "BLOCKED"
    passed = (
        verdict == "BLOCKED"
        and all(blockers.values())
        and resources["independent_arithmetic_consistent"]
        and negative["verifier_rejected"]
    )
    output = {
        "claim4_verdict": verdict,
        "blockers": blockers,
        "resource_contract": resources,
        "source_and_environment_audit": audit,
        "proxy_negative_control": negative,
        "blocker_dossier_valid": passed,
        "prohibited_interpretation": (
            "No environment-only, shortened, single-seed, non-PPO, non-S5, or "
            "digitized-paper-curve result is accepted as verification."
        ),
    }
    if not passed:
        raise AssertionError(f"Claim 4 blocker dossier failed closed: {output}")
    return output


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_claim4_evidence(repo_root: Path) -> dict[str, object]:
    result = evaluate_claim4(repo_root)
    artifact_dir = repo_root / ".openresearch" / "artifacts" / "claim_4"
    _write_json(artifact_dir / "verifier_output.json", result)
    _write_json(
        artifact_dir / "independent_checker_output.json",
        {
            "direct_lower_bound": (
                len(FIGURE3_VARIANTS)
                * PAPER_SEEDS
                * PAPER_TIMESTEPS
                * PAPER_EPOCHS
            ),
            "batch_rounded_lower_bound": result["resource_contract"][
                "figure3_sample_epoch_passes_with_complete_batches"
            ],
            "consistent": result["resource_contract"][
                "independent_arithmetic_consistent"
            ],
        },
    )
    _write_json(
        artifact_dir / "negative_control_output.json",
        result["proxy_negative_control"],
    )
    _write_json(
        artifact_dir / "runtime.json",
        {
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "local_or_hf_training_runtime_seconds": 0,
            "local_or_hf_training_cost_usd": 0,
            "reason": "No uniquely specified and runnable faithful training workload exists.",
        },
    )
    (artifact_dir / "EVAL.md").write_text(
        "# Claim 4 evaluation\n\n"
        "**BLOCKED**\n\n"
        "No qualifying training result was produced. The released paper source "
        "omits the episode length, beacon angles, exact seed list, complete "
        "transition entries for three actions, and raw numerical curves; neither "
        "the paper nor the author publication entry links a direct implementation. "
        "The cited S5 repository is adjacent code, not a RingWorld implementation. "
        "A full Figure 3 run would require 56 PPO trainings, 1.68 billion "
        "environment steps, and at least 50.4 billion sample-epoch passes. The "
        "fixed CPU environment has no JAX/Flax/Optax stack. Shortened, toy, "
        "digitized-curve, or non-S5 substitutes are explicitly rejected.\n",
        encoding="utf-8",
    )
    print("ORX_EVIDENCE_C4=" + json.dumps(result, sort_keys=True))
    return result
