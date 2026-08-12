"""Tests for the config-driven cloud sync in psdn_sonar.utils.data_downloader."""

import pytest

pytest.importorskip("boto3")

from psdn_sonar.utils.data_downloader import (  # noqa: E402
    DataDownloader,
    SyncConfig,
    SyncItem,
    sync_from_config,
)


def _write_config(tmp_path, text):
    path = tmp_path / "sync.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID_R2_CONFIG = """
provider: r2
bucket: my-bucket
prefix: deliveries/2026-01/
directories:
  - remote: audio/
    local: data/audio
files:
  - remote: manifest.jsonl
    local: data/metadata/manifest.jsonl
"""


class TestSyncConfigFromYaml:
    def test_parses_valid_config(self, tmp_path):
        cfg = SyncConfig.from_yaml(_write_config(tmp_path, VALID_R2_CONFIG))

        assert cfg.provider == "r2"
        assert cfg.bucket == "my-bucket"
        assert cfg.prefix == "deliveries/2026-01/"
        assert cfg.directories == [SyncItem(remote="audio/", local="data/audio")]
        assert cfg.files == [SyncItem(remote="manifest.jsonl", local="data/metadata/manifest.jsonl")]
        assert cfg.credential_envs["access_key_id"] == "R2_ACCESS_KEY_ID"
        assert cfg.credential_envs["account_id"] == "R2_ACCOUNT_ID"

    def test_s3_default_credential_envs(self, tmp_path):
        cfg = SyncConfig.from_yaml(
            _write_config(
                tmp_path,
                "provider: s3\nbucket: b\nfiles:\n  - remote: a.txt\n    local: a.txt\n",
            )
        )

        assert cfg.credential_envs["access_key_id"] == "AWS_ACCESS_KEY_ID"
        assert cfg.credential_envs["secret_access_key"] == "AWS_SECRET_ACCESS_KEY"

    def test_credential_env_overrides_with_env_suffix(self, tmp_path):
        cfg = SyncConfig.from_yaml(
            _write_config(
                tmp_path,
                VALID_R2_CONFIG + "credentials:\n  access_key_id_env: MY_KEY\n  account_id_env: MY_ACCOUNT\n",
            )
        )

        assert cfg.credential_envs["access_key_id"] == "MY_KEY"
        assert cfg.credential_envs["account_id"] == "MY_ACCOUNT"
        assert cfg.credential_envs["secret_access_key"] == "R2_SECRET_ACCESS_KEY"

    def test_unknown_credential_key_rejected(self, tmp_path):
        path = _write_config(tmp_path, VALID_R2_CONFIG + "credentials:\n  password_env: NOPE\n")

        with pytest.raises(ValueError, match="unknown credentials key"):
            SyncConfig.from_yaml(path)

    def test_unknown_provider_rejected(self, tmp_path):
        path = _write_config(tmp_path, "provider: gcs\nbucket: b\nfiles:\n  - remote: a\n    local: a\n")

        with pytest.raises(ValueError, match="provider must be one of"):
            SyncConfig.from_yaml(path)

    def test_missing_bucket_rejected(self, tmp_path):
        path = _write_config(tmp_path, "provider: s3\nfiles:\n  - remote: a\n    local: a\n")

        with pytest.raises(ValueError, match="bucket is required"):
            SyncConfig.from_yaml(path)

    def test_no_targets_rejected(self, tmp_path):
        path = _write_config(tmp_path, "provider: s3\nbucket: b\n")

        with pytest.raises(ValueError, match="at least one entry"):
            SyncConfig.from_yaml(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SyncConfig.from_yaml(tmp_path / "missing.yaml")


class TestCreateDownloader:
    def test_r2_endpoint_built_from_account_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("R2_ACCOUNT_ID", "acct123")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
        cfg = SyncConfig.from_yaml(_write_config(tmp_path, VALID_R2_CONFIG))

        downloader = cfg.create_downloader()

        assert downloader.s3_client.meta.endpoint_url == "https://acct123.r2.cloudflarestorage.com"

    def test_r2_without_account_id_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
        cfg = SyncConfig.from_yaml(_write_config(tmp_path, VALID_R2_CONFIG))

        with pytest.raises(ValueError, match="R2 requires"):
            cfg.create_downloader()

    def test_explicit_endpoint_used_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
        cfg = SyncConfig.from_yaml(
            _write_config(tmp_path, VALID_R2_CONFIG + "endpoint_url: https://storage.example.com\n")
        )

        downloader = cfg.create_downloader()

        assert downloader.s3_client.meta.endpoint_url == "https://storage.example.com"


class _FakeDownloader:
    def __init__(self):
        self.folder_calls = []
        self.file_calls = []
        self.fail_keys = set()

    def download_folder(self, bucket, prefix, local_dir, skip_existing=False):
        self.folder_calls.append((bucket, prefix, local_dir, skip_existing))
        return ["a.wav", "b.wav"]

    def download_file(self, bucket, key, local_path):
        self.file_calls.append((bucket, key, local_path))
        return key not in self.fail_keys


class TestSyncFromConfig:
    def _config(self):
        return SyncConfig(
            provider="r2",
            bucket="my-bucket",
            prefix="deliveries/",
            directories=[SyncItem(remote="audio/", local="data/audio")],
            files=[
                SyncItem(remote="manifest.jsonl", local="data/manifest.jsonl"),
                SyncItem(remote="missing.json", local="data/missing.json"),
            ],
        )

    def test_syncs_directories_and_files_with_prefix(self, monkeypatch):
        cfg = self._config()
        fake = _FakeDownloader()
        monkeypatch.setattr(SyncConfig, "create_downloader", lambda self: fake)

        summary = sync_from_config(cfg)

        assert fake.folder_calls == [("my-bucket", "deliveries/audio/", "data/audio", True)]
        assert [c[1] for c in fake.file_calls] == ["deliveries/manifest.jsonl", "deliveries/missing.json"]
        assert summary == {"downloaded": 4, "failed": []}

    def test_failed_file_targets_reported(self, monkeypatch):
        cfg = self._config()
        fake = _FakeDownloader()
        fake.fail_keys = {"deliveries/missing.json"}
        monkeypatch.setattr(SyncConfig, "create_downloader", lambda self: fake)

        summary = sync_from_config(cfg)

        assert summary["downloaded"] == 3
        assert summary["failed"] == ["deliveries/missing.json"]

    def test_accepts_config_path(self, tmp_path, monkeypatch):
        fake = _FakeDownloader()
        monkeypatch.setattr(SyncConfig, "create_downloader", lambda self: fake)
        path = _write_config(tmp_path, VALID_R2_CONFIG)

        summary = sync_from_config(path)

        assert summary["downloaded"] == 3


class _FakePaginator:
    def __init__(self, objects):
        self._objects = objects

    def paginate(self, Bucket, Prefix):
        return [{"Contents": self._objects}]


class TestDownloadFolderSkipExisting:
    def _downloader(self, tmp_path, objects):
        downloader = DataDownloader.__new__(DataDownloader)

        class _FakeClient:
            def get_paginator(self, name):
                return _FakePaginator(objects)

            def download_file(self, bucket, key, local_path):
                with open(local_path, "w") as f:
                    f.write("x" * 5)

        downloader.s3_client = _FakeClient()
        return downloader

    def test_same_size_file_skipped(self, tmp_path):
        objects = [{"Key": "audio/a.wav", "Size": 5}, {"Key": "audio/b.wav", "Size": 5}]
        downloader = self._downloader(tmp_path, objects)
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        (local_dir / "a.wav").write_text("x" * 5)

        fetched = downloader.download_folder("b", "audio/", str(local_dir), skip_existing=True)

        assert [p.split("/")[-1] for p in fetched] == ["b.wav"]

    def test_size_mismatch_redownloaded(self, tmp_path):
        objects = [{"Key": "audio/a.wav", "Size": 5}]
        downloader = self._downloader(tmp_path, objects)
        local_dir = tmp_path / "out"
        local_dir.mkdir()
        (local_dir / "a.wav").write_text("stale contents")

        fetched = downloader.download_folder("b", "audio/", str(local_dir), skip_existing=True)

        assert len(fetched) == 1
        assert (local_dir / "a.wav").read_text() == "x" * 5

    def test_directory_markers_ignored(self, tmp_path):
        objects = [{"Key": "audio/", "Size": 0}, {"Key": "audio/a.wav", "Size": 5}]
        downloader = self._downloader(tmp_path, objects)

        fetched = downloader.download_folder("b", "audio/", str(tmp_path / "out"), skip_existing=True)

        assert len(fetched) == 1
