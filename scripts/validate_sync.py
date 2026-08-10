#!/usr/bin/env python3
"""Validate a public-safe, tracked-files-only synchronization plan."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

ALLOWED_COMPONENTS = {
    "product-code": "psdn_sonar/",
    "public-config": "config/",
    "tests": "tests/",
}
ALLOWED_SYNC_TYPES = {"initial-import", "recurring-sync", "security-hotfix"}
ALLOWED_VALIDATIONS = {
    "Internal-reference gate",
    "Lint and type check",
    "Package artifacts",
    "Pre-commit baseline",
    "Secret scan",
    "Tests",
}
REQUIRED_VALIDATIONS = {
    "Internal-reference gate",
    "Lint and type check",
    "Pre-commit baseline",
    "Secret scan",
    "Tests",
}
REQUIRED_FIELDS = {
    "files",
    "import_gate",
    "included_components",
    "public_base_sha",
    "reviewer",
    "schema_version",
    "sync_date",
    "sync_type",
    "validation",
}
BLOCKED_PRODUCT_MODULES = {"cli_prep", "control_plane", "service"}
ALLOWED_PUBLIC_FILE_MODES = {b"100644", b"100755"}
SHA_RE = re.compile(r"[0-9a-f]{40}")
REVIEWER_RE = re.compile(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")


class SyncValidationError(ValueError):
    """Raised when a synchronization record is not safe or internally consistent."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class PublicSafeArgumentParser(argparse.ArgumentParser):
    """Avoid echoing malformed command-line values into public logs."""

    def error(self, message: str) -> None:
        del message
        self.exit(2, "Sync dry-run FAIL [invalid_arguments]\n")


def _load_record(path: Path) -> dict[str, Any]:
    try:
        raw_value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SyncValidationError("record_unreadable") from exc
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SyncValidationError("record_invalid_json") from exc
    if not isinstance(value, dict):
        raise SyncValidationError("record_not_object")
    return value


def _load_file_list(path: Path, category: str) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SyncValidationError(f"{category}_unreadable") from exc
    if not lines:
        raise SyncValidationError(f"{category}_empty")
    if any(not line or line != line.strip() or not line.isprintable() for line in lines):
        raise SyncValidationError(f"{category}_invalid")
    if len(lines) != len(set(lines)):
        raise SyncValidationError(f"{category}_duplicate")
    return set(lines)


def _expect_string_list(record: dict[str, Any], field: str) -> list[str]:
    value = record[field]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise SyncValidationError(f"{field}_invalid")
    if len(value) != len(set(value)):
        raise SyncValidationError(f"{field}_duplicate")
    return value


def _component_for_path(path: str) -> str:
    if path != path.strip() or not path.isprintable():
        raise SyncValidationError("path_invalid")
    if "\\" in path or any(character in path for character in "*?[]{}"):
        raise SyncValidationError("path_pattern")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path:
        raise SyncValidationError("path_unsafe")

    product_prefix = ALLOWED_COMPONENTS["product-code"]
    if path.startswith(product_prefix):
        first_part = path.removeprefix(product_prefix).split("/", maxsplit=1)[0]
        blocked_names = BLOCKED_PRODUCT_MODULES | {f"{name}.py" for name in BLOCKED_PRODUCT_MODULES}
        if first_part in blocked_names:
            raise SyncValidationError("path_excluded")

    for component, prefix in ALLOWED_COMPONENTS.items():
        if path.startswith(prefix) and len(path) > len(prefix):
            return component
    raise SyncValidationError("path_not_allowlisted")


def _validate_record(record: dict[str, Any], *, final_diff: bool) -> tuple[list[str], list[str]]:
    fields = set(record)
    if fields != REQUIRED_FIELDS:
        raise SyncValidationError("record_schema")

    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise SyncValidationError("schema_version_invalid")
    if not isinstance(record["sync_type"], str) or record["sync_type"] not in ALLOWED_SYNC_TYPES:
        raise SyncValidationError("sync_type_invalid")
    try:
        parsed_date = date.fromisoformat(record["sync_date"])
    except (TypeError, ValueError) as exc:
        raise SyncValidationError("sync_date_invalid") from exc
    if parsed_date.isoformat() != record["sync_date"]:
        raise SyncValidationError("sync_date_invalid")
    if not isinstance(record["public_base_sha"], str) or not SHA_RE.fullmatch(record["public_base_sha"]):
        raise SyncValidationError("public_base_sha_invalid")
    if not isinstance(record["reviewer"], str) or not REVIEWER_RE.fullmatch(record["reviewer"]):
        raise SyncValidationError("reviewer_invalid")
    expected_gate = "PASS" if final_diff else "PENDING"
    if record["import_gate"] != expected_gate:
        raise SyncValidationError("import_gate_phase_mismatch")

    files = _expect_string_list(record, "files")
    if files != sorted(files):
        raise SyncValidationError("files_unsorted")
    derived_components = {_component_for_path(path) for path in files}

    components = _expect_string_list(record, "included_components")
    if components != sorted(components):
        raise SyncValidationError("included_components_unsorted")
    if set(components) != derived_components:
        raise SyncValidationError("included_components_mismatch")

    validations = _expect_string_list(record, "validation")
    unknown_validations = set(validations) - ALLOWED_VALIDATIONS
    if unknown_validations:
        raise SyncValidationError("validation_unknown")
    missing_validations = REQUIRED_VALIDATIONS - set(validations)
    if missing_validations:
        raise SyncValidationError("validation_missing")

    return files, components


def _run_git(public_repo_path: Path, *arguments: str, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(public_repo_path), *arguments],
            check=False,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise SyncValidationError("public_repo_unavailable") from exc


def _git_paths(public_repo_path: Path, *arguments: str) -> set[str]:
    result = _run_git(public_repo_path, *arguments, capture=True)
    if result.returncode != 0:
        raise SyncValidationError("public_diff_unavailable")
    try:
        values = [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]
    except UnicodeError as exc:
        raise SyncValidationError("public_diff_invalid") from exc
    return set(values)


def _validate_public_repo(
    public_repo_path: Path,
    public_base_sha: str,
    *,
    final_diff: bool,
) -> set[str]:
    commit = _run_git(public_repo_path, "cat-file", "-e", f"{public_base_sha}^{{commit}}")
    if commit.returncode != 0:
        raise SyncValidationError("public_base_unknown")

    ancestor = _run_git(public_repo_path, "merge-base", "--is-ancestor", public_base_sha, "HEAD")
    if ancestor.returncode == 1:
        raise SyncValidationError("public_base_not_ancestor")
    if ancestor.returncode != 0:
        raise SyncValidationError("public_repo_unavailable")

    head = _run_git(public_repo_path, "rev-parse", "HEAD", capture=True)
    if head.returncode != 0:
        raise SyncValidationError("public_repo_unavailable")
    if not final_diff and head.stdout.decode("ascii", errors="ignore").strip() != public_base_sha:
        raise SyncValidationError("public_base_not_head")

    if _git_paths(public_repo_path, "ls-files", "--others", "--exclude-standard", "-z"):
        raise SyncValidationError("public_repo_untracked")

    unstaged = _run_git(public_repo_path, "diff", "--quiet", "--no-ext-diff", "--")
    if unstaged.returncode == 1:
        raise SyncValidationError("public_repo_unstaged")
    if unstaged.returncode != 0:
        raise SyncValidationError("public_diff_unavailable")

    changed_files = _git_paths(
        public_repo_path,
        "diff",
        "--cached",
        "--name-only",
        "--no-ext-diff",
        "--no-renames",
        "-z",
        public_base_sha,
        "--",
    )
    if not final_diff and changed_files:
        raise SyncValidationError("public_repo_dirty")
    if final_diff:
        if _git_paths(
            public_repo_path,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=DT",
            "--no-ext-diff",
            "--no-renames",
            "-z",
            public_base_sha,
            "--",
        ):
            raise SyncValidationError("public_diff_unsafe_change")
    return changed_files


def _validate_public_file_modes(public_repo_path: Path, selected_files: set[str]) -> None:
    result = _run_git(public_repo_path, "ls-files", "--stage", "-z", "--", *sorted(selected_files), capture=True)
    if result.returncode != 0:
        raise SyncValidationError("public_diff_unavailable")
    entries = [entry for entry in result.stdout.split(b"\0") if entry]
    if len(entries) != len(selected_files):
        raise SyncValidationError("public_diff_mismatch")
    for entry in entries:
        metadata, separator, _ = entry.partition(b"\t")
        mode = metadata.split(b" ", maxsplit=1)[0]
        if not separator or mode not in ALLOWED_PUBLIC_FILE_MODES:
            raise SyncValidationError("public_file_mode_unsafe")


def validate_sync(
    record_path: Path,
    source_files_path: Path,
    changed_files_path: Path | None = None,
    *,
    public_repo_path: Path,
) -> int:
    """Validate one sync record and return the number of allowlisted files."""

    record = _load_record(record_path)
    final_diff = changed_files_path is not None
    files, _ = _validate_record(record, final_diff=final_diff)
    selected_files = set(files)
    public_changed_files = _validate_public_repo(
        public_repo_path,
        record["public_base_sha"],
        final_diff=final_diff,
    )

    tracked_source_files = _load_file_list(source_files_path, "source_inventory")
    missing_from_source = selected_files - tracked_source_files
    if missing_from_source:
        raise SyncValidationError("source_inventory_missing_file")

    if changed_files_path is not None:
        changed_files = _load_file_list(changed_files_path, "changed_files")
        if changed_files != selected_files or public_changed_files != selected_files:
            raise SyncValidationError("public_diff_mismatch")
        _validate_public_file_modes(public_repo_path, selected_files)

    return len(files)


def _parse_args() -> argparse.Namespace:
    parser = PublicSafeArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path, help="Public-safe JSON sync record")
    parser.add_argument(
        "--source-tracked-files",
        required=True,
        type=Path,
        help="Temporary newline-delimited output from git ls-files in the canonical source",
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        help="Optional newline-delimited public diff; must exactly match the record",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        file_count = validate_sync(
            args.record,
            args.source_tracked_files,
            args.changed_files,
            public_repo_path=Path.cwd(),
        )
    except SyncValidationError as exc:
        print(f"Sync dry-run FAIL [{exc.code}]", file=sys.stderr)
        return 1
    except Exception:
        print("Sync dry-run FAIL [internal_error]", file=sys.stderr)
        return 1
    print(f"Sync dry-run PASS: {file_count} tracked allowlisted file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
