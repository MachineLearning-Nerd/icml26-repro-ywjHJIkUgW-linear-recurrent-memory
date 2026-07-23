"""Fail-closed validation for the cumulative logbook release candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HISTORICAL_HASHES = {
    "README.md": "843409f4206fd2e6528e0aa0d8d52eded567a7d7001dd14785d7cc2547130aa7",
    "bucket-icon.svg": "d1c28fc0a4e07f2688d013f576cf76ffc422d278d56a52a82989e0b93b3b3964",
    "index.html": "0920a62264fcdfef9ec48b48b35583b621c2ab8fca1f9e26158773b5b11a81f1",
    "logbook.css": "64e1de4358c79ec0d5f2697c56f98258c025e992c94ad7b3b7801739222ca41d",
    "logbook.js": "69d73869184f936613668569980f31984be65229e77c4df4ba9604d3de70c02b",
    "pages/claim-1-exact-belief-logit-reproduction/page.md":
        "36e743f51dc01526412d0f3bde8bdc1c3b3f5c22bc787fddb58e17c97ea01473",
    "pages/claim-2-vanishing-decoding-error/page.md":
        "840203d472fc182723d3fcc291835aab299f10b5b40ec8ad04e8506c8917707f",
    "pages/conclusion/page.md":
        "03ec0fc99049bd34be01ba2f9846dc3a402ca85ef1083677e6aece699d99a032",
    "pages/index.md":
        "767b8ac4f9ae27bcdaa66157eb0473d25e690bd3597d66a69f502d5bcb33d390",
    "pages/methods-environment/page.md":
        "f0a270d448c5fd2b5ffa800e155c07a1f701432452174c2511ec60fcdb765a96",
    "pages/negative-controls-falsification/page.md":
        "406ef77e1b225f5fc9fed3c8dc714c6c966f04dd386eecb5072e27e2dae08e71",
    "trackio-logo-light.png":
        "a6eb72253c0128ce79b526a86b7943eed37beec186b5f57ff6c1701d0e9ff596",
    "trackio-logo.png":
        "3e3792061d4d095759da30d7cfe7f14b621901793cd4d677b61b2896f5bf472b",
    "trackio-wordmark-dark.png":
        "71da94795855710d214801eb9b9b7b8898e9a8757abac0e22966a0531bbb2f4f",
}

HISTORICAL_SLUGS = {
    "claim-1-exact-belief-logit-reproduction",
    "claim-2-vanishing-decoding-error",
    "methods-environment",
    "negative-controls-falsification",
    "conclusion",
}

REQUIRED_CLAIM_FILES = {
    "claim_contract.json",
    "source_audit.md",
    "method.md",
    "limitations.md",
    "exact_command.txt",
    "EVAL.md",
    "verifier_output.json",
    "independent_checker_output.json",
    "negative_control_output.json",
    "runtime.json",
}

EXPECTED_VERDICTS = {
    "claim_1": "VERIFIED",
    "claim_2": "BLOCKED",
    "claim_3": "BLOCKED",
    "claim_4": "BLOCKED",
    "claim_5": "VERIFIED",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(node: dict) -> list[dict]:
    result = [node]
    for child in node.get("children", []):
        result.extend(_flatten(child))
    return result


def validate_release_candidate(repo_root: Path) -> dict:
    logbook_root = repo_root / ".trackio" / "logbook"
    manifest = json.loads((logbook_root / "logbook.json").read_text())
    nodes = _flatten(manifest["root"])
    slugs = [node["slug"] for node in nodes]
    if len(slugs) != len(set(slugs)):
        raise AssertionError("logbook contains duplicate slugs")
    for node in nodes:
        if not (logbook_root / node["file"]).is_file():
            raise AssertionError(f"missing logbook page: {node['file']}")

    historical_hashes_match = {
        path: _sha256(logbook_root / path) == expected
        for path, expected in HISTORICAL_HASHES.items()
    }
    if not all(historical_hashes_match.values()):
        raise AssertionError("a protected historical page or asset changed")
    present_slugs = set(slugs)
    if not HISTORICAL_SLUGS <= present_slugs:
        raise AssertionError("historical logbook entries are not all reachable")

    claim_checks = {}
    for claim, verdict in EXPECTED_VERDICTS.items():
        directory = logbook_root / "evidence" / claim
        missing = sorted(
            filename
            for filename in REQUIRED_CLAIM_FILES
            if not (directory / filename).is_file()
        )
        if missing:
            raise AssertionError(f"{claim} missing release evidence: {missing}")
        eval_text = (directory / "EVAL.md").read_text()
        if f"**{verdict}**" not in eval_text:
            raise AssertionError(f"{claim} does not declare {verdict}")
        contract = json.loads((directory / "claim_contract.json").read_text())
        verifier = json.loads((directory / "verifier_output.json").read_text())
        claim_checks[claim] = {
            "contract_json_valid": bool(contract),
            "verifier_json_valid": bool(verifier),
            "expected_verdict": verdict,
        }

    result = {
        "candidate_logbook_valid": True,
        "space_id": manifest["space_id"],
        "historical_file_count_checked": len(HISTORICAL_HASHES),
        "historical_files_hash_identical": all(historical_hashes_match.values()),
        "historical_slugs_reachable": sorted(HISTORICAL_SLUGS),
        "logbook_page_count": len(nodes),
        "claim_checks": claim_checks,
    }
    print("RELEASE_CHECK " + json.dumps(result, sort_keys=True))
    return result
