"""Sync a remote dataset to local disk from a YAML config.

Thin CLI over :func:`psdn_sonar.utils.data_downloader.sync_from_config`.
The config declares the provider (S3 or R2), bucket, and remote-to-local
mappings; credentials are always read from environment variables (a
``.env`` file next to the config is loaded if present). See
``scripts/download_config.example.yaml`` for the full schema.

Requires the ``cloud`` extra (``boto3``).
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from psdn_sonar.utils.data_downloader import sync_from_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a dataset from S3/R2 as declared in a YAML config")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML sync config")
    args = parser.parse_args()

    for env_path in (args.config.parent / ".env", Path.cwd() / ".env"):
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            break
    else:
        load_dotenv()

    try:
        summary = sync_from_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Downloaded %d files", summary["downloaded"])
    if summary["failed"]:
        logger.error("Failed to download: %s", ", ".join(summary["failed"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
