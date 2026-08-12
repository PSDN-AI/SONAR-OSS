"""Download evaluation datasets from S3-compatible object storage.

Snippets for fetching single files or full manifest-based datasets from
AWS S3 or Cloudflare R2. Requires the ``cloud`` extra
(``pip install "psdn-sonar[cloud]"``) and your own credentials.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def example_s3_single_file():
    from psdn_sonar.utils.data_downloader import download_from_s3

    success = download_from_s3(
        bucket_name="my-asr-bucket",
        object_key="datasets/bengali/manifest.jsonl",
        local_path="./data/manifest.jsonl",
        aws_access_key_id="YOUR_ACCESS_KEY",
        aws_secret_access_key="YOUR_SECRET_KEY",
        region_name="us-east-1",
    )
    logger.info("Download %s", "succeeded" if success else "failed")


def example_r2_single_file():
    from psdn_sonar.utils.data_downloader import download_from_r2

    success = download_from_r2(
        bucket_name="my-asr-bucket",
        object_key="datasets/bengali/manifest.jsonl",
        local_path="./data/manifest.jsonl",
        account_id="YOUR_CLOUDFLARE_ACCOUNT_ID",
        access_key_id="YOUR_R2_ACCESS_KEY",
        secret_access_key="YOUR_R2_SECRET_KEY",
    )
    logger.info("Download %s", "succeeded" if success else "failed")


def example_full_dataset_s3():
    from psdn_sonar.utils.data_downloader import download_dataset_from_cloud

    tsv_path, audio_files = download_dataset_from_cloud(
        bucket_name="my-asr-bucket",
        manifest_key="datasets/bengali/manifest.jsonl",
        audio_prefix="datasets/bengali/audio",
        output_dir="./data/bengali",
        cloud_type="s3",
        access_key_id="YOUR_ACCESS_KEY",
        secret_access_key="YOUR_SECRET_KEY",
    )
    logger.info("Dataset ready: %s (%d audio files)", tsv_path, len(audio_files))


def example_full_dataset_r2():
    from psdn_sonar.utils.data_downloader import download_dataset_from_cloud

    tsv_path, audio_files = download_dataset_from_cloud(
        bucket_name="my-asr-bucket",
        manifest_key="datasets/bengali/manifest.jsonl",
        audio_prefix="datasets/bengali/audio",
        output_dir="./data/bengali",
        cloud_type="r2",
        account_id="YOUR_CLOUDFLARE_ACCOUNT_ID",
        access_key_id="YOUR_R2_ACCESS_KEY",
        secret_access_key="YOUR_R2_SECRET_KEY",
    )
    logger.info("Dataset ready: %s (%d audio files)", tsv_path, len(audio_files))


if __name__ == "__main__":
    logger.info("Example usage — update credentials, then call one of:")
    logger.info("  example_s3_single_file()")
    logger.info("  example_r2_single_file()")
    logger.info("  example_full_dataset_s3()")
    logger.info("  example_full_dataset_r2()")
