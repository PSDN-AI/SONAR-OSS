import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.validate_sync as sync_validator
from scripts.validate_sync import SyncValidationError, validate_sync

DEFAULT_FILES = ["psdn_sonar/example.py", "tests/unit/test_example.py"]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_public_file(repo: Path, relative_path: str, content: str) -> None:
    target = repo / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _init_public_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "public"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Sync Policy Test")
    _git(repo, "config", "user.email", "sync-policy@example.com")
    for relative_path in DEFAULT_FILES:
        _write_public_file(repo, relative_path, "baseline\n")
    _git(repo, "add", "--", *DEFAULT_FILES)
    _git(repo, "commit", "--quiet", "-m", "baseline")
    parent_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "current public base")
    return repo, _git(repo, "rev-parse", "HEAD"), parent_sha


def _record(public_base_sha: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sync_type": "recurring-sync",
        "sync_date": "2026-08-10",
        "public_base_sha": public_base_sha,
        "included_components": ["product-code", "tests"],
        "files": DEFAULT_FILES.copy(),
        "reviewer": "@reviewer",
        "validation": [
            "Internal-reference gate",
            "Lint and type check",
            "Pre-commit baseline",
            "Secret scan",
            "Tests",
        ],
        "import_gate": "PASS",
    }


def _case(
    tmp_path: Path,
    *,
    record_updates: dict[str, object] | None = None,
    source: list[str] | None = None,
    changed: list[str] | None = None,
    actual_changed: list[str] | None = None,
    deleted: set[str] | None = None,
    stage_changes: bool = True,
) -> tuple[Path, Path, Path | None, Path, str]:
    repo, public_base_sha, parent_sha = _init_public_repo(tmp_path)
    record_value = _record(public_base_sha)
    if changed is None:
        record_value["import_gate"] = "PENDING"
    if record_updates:
        record_value.update(record_updates)

    actual_paths = changed if actual_changed is None else actual_changed
    deleted = deleted or set()
    if actual_paths is not None:
        for relative_path in actual_paths:
            target = repo / relative_path
            if relative_path in deleted:
                target.unlink()
            else:
                _write_public_file(repo, relative_path, "synchronized\n")
        if stage_changes:
            _git(repo, "add", "--all", "--", *actual_paths)

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    record_path = inputs / "record.json"
    source_path = inputs / "source-files.txt"
    changed_path = inputs / "changed-files.txt"
    record_path.write_text(json.dumps(record_value), encoding="utf-8")
    source_values = list(record_value["files"]) if source is None else source
    source_path.write_text("\n".join(source_values) + "\n", encoding="utf-8")
    if changed is not None:
        changed_path.write_text("\n".join(changed) + "\n", encoding="utf-8")
        return record_path, source_path, changed_path, repo, parent_sha
    return record_path, source_path, None, repo, parent_sha


def _assert_error(code: str, record: Path, source: Path, changed: Path | None, repo: Path) -> None:
    with pytest.raises(SyncValidationError) as raised:
        validate_sync(record, source, changed, public_repo_path=repo)
    assert raised.value.code == code


def test_sync_dry_run_accepts_exact_tracked_allowlist(tmp_path):
    record, source, changed, repo, _ = _case(tmp_path, changed=DEFAULT_FILES)

    assert validate_sync(record, source, changed, public_repo_path=repo) == 2


def test_sync_dry_run_accepts_clean_pre_copy_check(tmp_path):
    record, source, changed, repo, _ = _case(tmp_path)

    assert validate_sync(record, source, changed, public_repo_path=repo) == 2


def test_sync_dry_run_rejects_untracked_selected_file(tmp_path):
    record, source, changed, repo, _ = _case(tmp_path, source=[DEFAULT_FILES[0]])

    _assert_error("source_inventory_missing_file", record, source, changed, repo)


def test_sync_dry_run_rejects_diff_outside_reviewed_allowlist(tmp_path):
    changed_files = [DEFAULT_FILES[0], "tests/unit/test_other.py"]
    record, source, changed, repo, _ = _case(tmp_path, changed=changed_files)

    _assert_error("public_diff_mismatch", record, source, changed, repo)


def test_sync_dry_run_rejects_stale_changed_file_evidence(tmp_path):
    actual_changed = [*DEFAULT_FILES, "tests/unit/test_other.py"]
    record, source, changed, repo, _ = _case(
        tmp_path,
        changed=DEFAULT_FILES,
        actual_changed=actual_changed,
    )

    _assert_error("public_diff_mismatch", record, source, changed, repo)


def test_sync_dry_run_rejects_duplicate_changed_file_evidence(tmp_path):
    changed_files = [DEFAULT_FILES[0], DEFAULT_FILES[0], DEFAULT_FILES[1]]
    record, source, changed, repo, _ = _case(
        tmp_path,
        changed=changed_files,
        actual_changed=DEFAULT_FILES,
    )

    _assert_error("changed_files_duplicate", record, source, changed, repo)


def test_sync_dry_run_rejects_unstaged_new_public_file(tmp_path):
    new_file = "psdn_sonar/new_file.py"
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"files": [new_file], "included_components": ["product-code"]},
        changed=[new_file],
        stage_changes=False,
    )

    _assert_error("public_repo_untracked", record, source, changed, repo)


def test_sync_dry_run_rejects_unstaged_tracked_change(tmp_path):
    record, source, changed, repo, _ = _case(
        tmp_path,
        changed=DEFAULT_FILES,
        stage_changes=False,
    )

    _assert_error("public_repo_unstaged", record, source, changed, repo)


def test_sync_dry_run_rejects_deletion_disguised_as_sync(tmp_path):
    deleted_file = DEFAULT_FILES[0]
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"files": [deleted_file], "included_components": ["product-code"]},
        changed=[deleted_file],
        deleted={deleted_file},
    )

    _assert_error("public_diff_unsafe_change", record, source, changed, repo)


def test_sync_dry_run_rejects_new_symlink(tmp_path):
    new_file = "psdn_sonar/new_file.py"
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"files": [new_file], "included_components": ["product-code"]},
        changed=[new_file],
        stage_changes=False,
    )
    target = repo / new_file
    target.unlink()
    target.symlink_to("/nonpublic/source.py")
    _git(repo, "add", "--", new_file)

    _assert_error("public_file_mode_unsafe", record, source, changed, repo)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (["README.md"], "path_not_allowlisted"),
        (["psdn_sonar/" + "service/api.py"], "path_excluded"),
        (["psdn_sonar/" + "service.py"], "path_excluded"),
        (["psdn_sonar/" + "cli_prep.py"], "path_excluded"),
        (["psdn_sonar/*.py"], "path_pattern"),
        (["psdn_sonar/./example.py"], "path_unsafe"),
        (["psdn_sonar//example.py"], "path_unsafe"),
        (["psdn_sonar/../README.md"], "path_unsafe"),
    ],
)
def test_sync_dry_run_rejects_unsafe_file_paths(tmp_path, value, code):
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"files": value, "included_components": ["product-code"]},
    )

    _assert_error(code, record, source, changed, repo)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("reviewer", "pending", "reviewer_invalid"),
        ("schema_version", True, "schema_version_invalid"),
        ("sync_type", [], "sync_type_invalid"),
        (
            "validation",
            ["Internal-reference gate", "Pre-commit baseline", "Secret scan"],
            "validation_missing",
        ),
    ],
)
def test_sync_dry_run_rejects_unsafe_record_values(tmp_path, field, value, code):
    record, source, changed, repo, _ = _case(tmp_path, record_updates={field: value})

    _assert_error(code, record, source, changed, repo)


def test_sync_dry_run_rejects_pending_import_gate_for_final_diff(tmp_path):
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"import_gate": "PENDING"},
        changed=DEFAULT_FILES,
    )

    _assert_error("import_gate_phase_mismatch", record, source, changed, repo)


def test_sync_dry_run_rejects_premature_pass_before_copy(tmp_path):
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"import_gate": "PASS"},
    )

    _assert_error("import_gate_phase_mismatch", record, source, changed, repo)


def test_sync_dry_run_rejects_private_source_metadata_field(tmp_path):
    record, source, changed, repo, _ = _case(tmp_path)
    record_value = json.loads(record.read_text(encoding="utf-8"))
    record_value["source_repository"] = "private-name"
    record.write_text(json.dumps(record_value), encoding="utf-8")

    _assert_error("record_schema", record, source, changed, repo)


def test_sync_dry_run_rejects_unknown_public_base(tmp_path):
    record, source, changed, repo, _ = _case(
        tmp_path,
        record_updates={"public_base_sha": "b" * 40},
    )

    _assert_error("public_base_unknown", record, source, changed, repo)


def test_sync_dry_run_requires_head_as_pre_copy_base(tmp_path):
    record, source, changed, repo, parent_sha = _case(tmp_path)
    record_value = json.loads(record.read_text(encoding="utf-8"))
    record_value["public_base_sha"] = parent_sha
    record.write_text(json.dumps(record_value), encoding="utf-8")

    _assert_error("public_base_not_head", record, source, changed, repo)


@pytest.mark.parametrize(
    "private_marker",
    ["private-repository-name", "customer-secret-path"],
)
def test_cli_failure_does_not_echo_private_record_values(tmp_path, monkeypatch, capsys, private_marker):
    private_path = f"/{private_marker}/source.py"
    record, source, _, repo, _ = _case(
        tmp_path,
        record_updates={"files": [private_path], "included_components": ["product-code"]},
    )
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_sync.py", "--record", str(record), "--source-tracked-files", str(source)],
    )

    assert sync_validator.main() == 1
    output = capsys.readouterr()
    assert private_marker not in output.out + output.err
    assert output.err == "Sync dry-run FAIL [path_unsafe]\n"


def test_cli_failure_does_not_echo_private_input_path(tmp_path, monkeypatch, capsys):
    private_marker = "private-source-checkout"
    record, _, _, repo, _ = _case(tmp_path)
    missing_source = tmp_path / private_marker / "tracked-files.txt"
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_sync.py", "--record", str(record), "--source-tracked-files", str(missing_source)],
    )

    assert sync_validator.main() == 1
    output = capsys.readouterr()
    assert private_marker not in output.out + output.err
    assert output.err == "Sync dry-run FAIL [source_inventory_unreadable]\n"


def test_cli_success_uses_actual_public_diff(tmp_path, monkeypatch, capsys):
    record, source, changed, repo, _ = _case(tmp_path, changed=DEFAULT_FILES)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_sync.py",
            "--record",
            str(record),
            "--source-tracked-files",
            str(source),
            "--changed-files",
            str(changed),
        ],
    )

    assert sync_validator.main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == "Sync dry-run PASS: 2 tracked allowlisted file(s)\n"


def test_cli_unexpected_failure_does_not_echo_private_details(monkeypatch, capsys):
    private_marker = "private-runtime-detail"

    def fail_validation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(private_marker)

    monkeypatch.setattr(sync_validator, "validate_sync", fail_validation)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_sync.py", "--record", "record.json", "--source-tracked-files", "source.txt"],
    )

    assert sync_validator.main() == 1
    output = capsys.readouterr()
    assert private_marker not in output.out + output.err
    assert output.err == "Sync dry-run FAIL [internal_error]\n"


def test_cli_argument_failure_does_not_echo_private_value(monkeypatch, capsys):
    private_marker = "private-repository-name"
    monkeypatch.setattr(sys, "argv", ["validate_sync.py", f"--{private_marker}"])

    with pytest.raises(SystemExit) as raised:
        sync_validator.main()

    assert raised.value.code == 2
    output = capsys.readouterr()
    assert private_marker not in output.out + output.err
    assert output.err == "Sync dry-run FAIL [invalid_arguments]\n"
