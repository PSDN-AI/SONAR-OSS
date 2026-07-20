"""Dataset download helpers for S3-compatible object storage.

Wraps ``boto3`` for the common evaluation workflow: fetch a manifest
(JSONL), convert it to the toolkit's TSV input format, and mirror the
referenced audio files locally. Works against AWS S3 or any S3-compatible
endpoint (e.g. Cloudflare R2) — credentials and bucket names are always
supplied by the caller, never hardcoded.

Requires the ``[cloud]`` extra (``boto3``).
"""

import logging
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads files and folders from an S3-compatible bucket."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "auto",
    ):
        """Create an S3 client.

        Args:
            endpoint_url: Custom endpoint for S3-compatible providers;
                ``None`` uses AWS S3.
            aws_access_key_id: Access key; ``None`` falls back to the
                standard boto3 credential chain (env vars, profile, role).
            aws_secret_access_key: Secret key; same fallback as above.
            region_name: Bucket region ("auto" for R2-style endpoints).
        """
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def download_file(self, bucket_name: str, object_key: str, local_path: str) -> bool:
        """Download a single object to ``local_path``, creating parent dirs.

        Returns ``True`` on success, ``False`` on client/credential errors
        (logged, not raised) so batch callers can tally partial failures.
        """
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading s3://{bucket_name}/{object_key} to {target}")
            self.s3_client.download_file(bucket_name, object_key, str(target))
            logger.info("Downloaded successfully")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to download: {e}")
            return False

    def download_folder(
        self, bucket_name: str, prefix: str, local_dir: str, file_extensions: Optional[List[str]] = None
    ) -> List[str]:
        """Download every object under ``prefix`` into ``local_dir``.

        Args:
            bucket_name: Source bucket.
            prefix: Key prefix to mirror; the prefix itself is stripped
                from local paths.
            local_dir: Destination directory (created if missing).
            file_extensions: If given, only keys ending in one of these
                extensions are downloaded.

        Returns:
            Local paths of successfully downloaded files. On listing
            errors the partial list downloaded so far is returned.
        """
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]

                    if file_extensions:
                        if not any(key.endswith(ext) for ext in file_extensions):
                            continue

                    relative_path = Path(key).relative_to(prefix) if prefix else Path(key)
                    local_file = target_dir / relative_path

                    if self.download_file(bucket_name, key, str(local_file)):
                        downloaded_files.append(str(local_file))

            logger.info(f"Downloaded {len(downloaded_files)} files to {target_dir}")
            return downloaded_files

        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to list/download folder: {e}")
            return downloaded_files

    def download_and_convert_to_tsv(
        self, bucket_name: str, manifest_key: str, output_tsv: str, audio_prefix: Optional[str] = None
    ) -> bool:
        """Download a JSONL manifest and convert it to the TSV input format.

        Accepts ``audio_path``/``audio_filepath`` and ``transcription``/
        ``transcript``/``text`` field spellings; rows missing either field
        are skipped. When ``audio_prefix`` is given it is prepended to each
        audio path so the TSV points at the locally mirrored files.
        """
        import json

        temp_manifest = Path(output_tsv).parent / "temp_manifest.jsonl"

        if not self.download_file(bucket_name, manifest_key, str(temp_manifest)):
            return False

        try:
            data = []
            with open(temp_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    audio_path = entry.get("audio_path") or entry.get("audio_filepath")
                    transcript = entry.get("transcription") or entry.get("transcript") or entry.get("text")

                    if audio_path and transcript:
                        if audio_prefix:
                            audio_path = f"{audio_prefix}/{audio_path}"
                        data.append({"audio_path": audio_path, "transcription": transcript})

            df = pd.DataFrame(data)
            df.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")

            temp_manifest.unlink()

            logger.info(f"Converted manifest to TSV: {output_tsv} ({len(df)} samples)")
            return True

        except Exception as e:
            logger.error(f"Failed to convert manifest: {e}")
            if temp_manifest.exists():
                temp_manifest.unlink()
            return False


def download_from_s3(
    bucket_name: str,
    object_key: str,
    local_path: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: str = "us-east-1",
) -> bool:
    """One-shot download of a single object from AWS S3."""
    downloader = DataDownloader(
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name
    )
    return downloader.download_file(bucket_name, object_key, local_path)


def download_from_r2(
    bucket_name: str, object_key: str, local_path: str, account_id: str, access_key_id: str, secret_access_key: str
) -> bool:
    """One-shot download of a single object from Cloudflare R2."""
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    downloader = DataDownloader(
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )
    return downloader.download_file(bucket_name, object_key, local_path)


def download_dataset_from_cloud(
    bucket_name: str,
    manifest_key: str,
    audio_prefix: str,
    output_dir: str,
    cloud_type: str = "s3",
    endpoint_url: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    account_id: Optional[str] = None,
) -> tuple[str, List[str]]:
    """Download a full dataset (manifest + audio) from S3 or R2.

    Produces ``<output_dir>/dataset.tsv`` referencing audio mirrored under
    ``<output_dir>/audio/``, ready to pass to the evaluators.

    Returns:
        Tuple of (path to the generated TSV, list of downloaded audio files).

    Raises:
        ValueError: ``cloud_type="r2"`` without an ``account_id``.
        RuntimeError: The manifest download or conversion failed.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cloud_type == "r2":
        if not account_id:
            raise ValueError("account_id required for R2")
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    downloader = DataDownloader(
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto" if cloud_type == "r2" else "us-east-1",
    )

    tsv_path = out_dir / "dataset.tsv"
    audio_dir = out_dir / "audio"

    logger.info("Step 1: Downloading and converting manifest to TSV")
    if not downloader.download_and_convert_to_tsv(
        bucket_name, manifest_key, str(tsv_path), audio_prefix=str(audio_dir)
    ):
        raise RuntimeError("Failed to download manifest")

    logger.info("Step 2: Downloading audio files")
    audio_files = downloader.download_folder(
        bucket_name, audio_prefix, str(audio_dir), file_extensions=[".wav", ".mp3", ".flac", ".m4a"]
    )

    logger.info(f"Dataset ready: {tsv_path} with {len(audio_files)} audio files")
    return str(tsv_path), audio_files
