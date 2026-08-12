#!/usr/bin/env python3
"""Download an ASR evaluation dataset (manifest + audio) from S3 or R2.

Fetches a JSONL manifest and its referenced audio files, converting the
manifest to the toolkit's TSV input format. Requires the ``cloud`` extra.
"""

import argparse
import logging
import os
import sys

from psdn_sonar.utils.data_downloader import download_dataset_from_cloud

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download ASR evaluation dataset from S3 or R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Download from AWS S3
  python scripts/download_dataset.py \\
    --cloud-type s3 \\
    --bucket my-asr-data \\
    --manifest-key datasets/bengali/manifest.jsonl \\
    --audio-prefix datasets/bengali/audio \\
    --output-dir ./data/bengali \\
    --access-key-id $AWS_ACCESS_KEY_ID \\
    --secret-access-key $AWS_SECRET_ACCESS_KEY

  # Download from Cloudflare R2
  python scripts/download_dataset.py \\
    --cloud-type r2 \\
    --bucket my-asr-data \\
    --manifest-key datasets/bengali/manifest.jsonl \\
    --audio-prefix datasets/bengali/audio \\
    --output-dir ./data/bengali \\
    --account-id YOUR_ACCOUNT_ID \\
    --access-key-id $R2_ACCESS_KEY_ID \\
    --secret-access-key $R2_SECRET_ACCESS_KEY
        """,
    )

    parser.add_argument("--cloud-type", choices=["s3", "r2"], default="s3", help="Cloud storage type (default: s3)")
    parser.add_argument("--bucket", required=True, help="Bucket name")
    parser.add_argument("--manifest-key", required=True, help="Path to manifest file in bucket (JSONL format)")
    parser.add_argument("--audio-prefix", required=True, help="Prefix/folder for audio files in bucket")
    parser.add_argument("--output-dir", required=True, help="Local directory to save dataset")
    parser.add_argument("--access-key-id", help="Access key ID (or set AWS_ACCESS_KEY_ID / R2_ACCESS_KEY_ID env var)")
    parser.add_argument(
        "--secret-access-key", help="Secret access key (or set AWS_SECRET_ACCESS_KEY / R2_SECRET_ACCESS_KEY env var)"
    )
    parser.add_argument("--account-id", help="Cloudflare account ID (required for R2)")
    parser.add_argument("--endpoint-url", help="Custom S3-compatible endpoint URL")

    args = parser.parse_args()

    access_key_id = args.access_key_id or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = (
        args.secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
    )

    if not access_key_id or not secret_access_key:
        logger.error(
            "Access credentials not provided. Set --access-key-id and --secret-access-key or environment variables."
        )
        sys.exit(1)

    if args.cloud_type == "r2" and not args.account_id:
        logger.error("--account-id required for Cloudflare R2")
        sys.exit(1)

    try:
        logger.info(
            "Downloading from %s: bucket=%s manifest=%s audio=%s -> %s",
            args.cloud_type.upper(),
            args.bucket,
            args.manifest_key,
            args.audio_prefix,
            args.output_dir,
        )

        tsv_path, audio_files = download_dataset_from_cloud(
            bucket_name=args.bucket,
            manifest_key=args.manifest_key,
            audio_prefix=args.audio_prefix,
            output_dir=args.output_dir,
            cloud_type=args.cloud_type,
            endpoint_url=args.endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            account_id=args.account_id,
        )

        logger.info("Download complete: %s (%d audio files)", tsv_path, len(audio_files))
        logger.info("Next: psdn-sonar single --input %s --models <model>", tsv_path)

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
