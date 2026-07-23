"""Build a text-only HF candidate overlay and reproducible release manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from release_checks import HISTORICAL_SLUGS, validate_release_candidate


JUDGED_REVISION = "e01d21cd25f3275204195770be8a25e3e274a734"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(node: dict) -> list[dict]:
    result = [node]
    for child in node.get("children", []):
        result.extend(flatten(child))
    return result


def prepare(repo_root: Path, protected_root: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output_root}")
    logbook_root = repo_root / ".trackio" / "logbook"
    validation = validate_release_candidate(repo_root)

    protected_files = sorted(
        path.relative_to(protected_root).as_posix()
        for path in protected_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    candidate_source_files = sorted(
        path.relative_to(logbook_root).as_posix()
        for path in logbook_root.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )
    protected_hashes = {
        path: sha256(protected_root / path) for path in protected_files
    }
    changed_or_new = [
        path
        for path in candidate_source_files
        if path not in protected_hashes
        or sha256(logbook_root / path) != protected_hashes[path]
    ]

    new_manifest = json.loads((logbook_root / "logbook.json").read_text())
    new_slugs = {node["slug"] for node in flatten(new_manifest["root"])}
    semantic_old_tree_subset = HISTORICAL_SLUGS <= new_slugs
    protected_content_unchanged = {
        path: sha256(logbook_root / path) == protected_hashes[path]
        for path in protected_files
        if path != "logbook.json" and (logbook_root / path).is_file()
    }
    subset = {
        "judged_revision": JUDGED_REVISION,
        "protected_file_count": len(protected_files),
        "protected_path_set_is_candidate_subset": True,
        "protected_paths": protected_files,
        "historical_logbook_tree_semantic_subset": semantic_old_tree_subset,
        "historical_pages_and_bundled_assets_hash_identical":
            all(protected_content_unchanged.values()),
        "hash_identical_checked_paths": sorted(protected_content_unchanged),
        "expected_additive_manifest_change": "logbook.json",
    }
    if not semantic_old_tree_subset or not all(protected_content_unchanged.values()):
        raise AssertionError("protected historical logbook evidence was not preserved")

    release_dir = logbook_root / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "protected_subset_check.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n"
    )
    (release_dir / "logbook_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    changed_or_new.extend(
        [
            "release/logbook_validation.json",
            "release/protected_subset_check.json",
            "release/UPLOAD_ALLOWLIST.txt",
            "release/SHA256SUMS.txt",
        ]
    )
    allowlist = sorted(set(changed_or_new))
    (release_dir / "UPLOAD_ALLOWLIST.txt").write_text(
        "\n".join(allowlist) + "\n"
    )
    sums_paths = [path for path in allowlist if path != "release/SHA256SUMS.txt"]
    sums = [
        f"{sha256(logbook_root / path)}  {path}"
        for path in sums_paths
    ]
    (release_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")

    secret_match_count = 0
    for relative in allowlist:
        data = (logbook_root / relative).read_bytes()
        secret_match_count += sum(bool(pattern.search(data)) for pattern in SECRET_PATTERNS)
    if secret_match_count:
        raise AssertionError("secret-like content found in upload allowlist")

    shutil.copytree(
        protected_root,
        output_root,
        copy_function=shutil.copyfile,
        ignore=shutil.ignore_patterns(".git"),
    )
    output_root.chmod(output_root.stat().st_mode | 0o700)
    for copied in output_root.rglob("*"):
        copied.chmod(copied.stat().st_mode | (0o700 if copied.is_dir() else 0o600))
    for source in logbook_root.rglob("*"):
        if source.is_file():
            relative = source.relative_to(logbook_root)
            target = output_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    final_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if not set(protected_files) <= final_paths:
        raise AssertionError("protected path set is not a subset of candidate")

    result = {
        "candidate_root": str(output_root),
        "judged_revision": JUDGED_REVISION,
        "old_path_subset_count": len(protected_files),
        "upload_file_count": len(allowlist),
        "upload_allowlist": allowlist,
        "secret_pattern_matches": secret_match_count,
        "text_only_upload": all(
            (logbook_root / path).suffix in TEXT_SUFFIXES for path in allowlist
        ),
    }
    print("RELEASE_PACKAGE " + json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.repo_root.resolve(),
        args.protected_root.resolve(),
        args.output_root.resolve(),
    )


if __name__ == "__main__":
    main()
