import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from upstash_redis import Redis

from src.data.handlers.redis_log_handler import RedisLogHandler
from src.data.handlers.s3_handler import S3Handler
from src.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)

BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
AWS_REGION: str = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "")

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))
timestamp = datetime.now()
redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)


class Config:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.use_minio: bool = config["use_minio"]
        self.batch_size: int = config["redis"]["batch_size"]
        self.interaction_prefix: str = config["redis"]["interaction_prefix"]
        self.migration_marker: str = config["redis"]["migration_marker"]
        self.safety_margin_ms: int = config["redis"]["safety_margin_ms"]


def get_s3_handler(config: Config) -> S3Handler:
    if config.use_minio:
        return S3Handler(
            aws_region=AWS_REGION,
            bucket_name=BUCKET_NAME,
            endpoint_url=MINIO_ENDPOINT,
            verify_ssl=False,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )
    return S3Handler(
        aws_region=AWS_REGION,
        bucket_name=BUCKET_NAME,
    )


def main() -> None:
    try:
        config: Config = Config(config_dict)
        s3_handler: S3Handler = get_s3_handler(config)

        redis_handler = RedisLogHandler(redis_client)

        last_processed_ts = redis_handler.get_last_processed_timestamp(
            config.migration_marker
        )

        current_time_ms = int(timestamp.timestamp() * 1000)
        end_ts = current_time_ms - config.safety_margin_ms

        logger.info("Processing logs from %d to %d", last_processed_ts, end_ts)

        interaction_keys = redis_handler.get_interaction_keys(
            config.interaction_prefix,
            config.batch_size,
            last_processed_ts,
            end_ts,
        )
        logger.info("Found %d interaction logs to process", len(interaction_keys))

        total_processed = 0

        for i in range(0, len(interaction_keys), config.batch_size):
            batch_keys = interaction_keys[i : i + config.batch_size]
            logger.info(
                "Processing batch %d, size: %d",
                i // config.batch_size + 1,
                len(batch_keys),
            )

            logs = redis_handler.fetch_interaction_logs(batch_keys)
            results = redis_handler.save_and_delete_interaction_logs(
                s3_handler, logs, batch_keys, timestamp
            )

            batch_processed = sum(results.values())
            total_processed += batch_processed
            logger.info(
                "Processed %d logs in this batch, total: %d",
                batch_processed,
                total_processed,
            )

        if interaction_keys:
            redis_handler.update_last_processed_timestamp(
                config.migration_marker, end_ts
            )
            logger.info("Updated last processed timestamp to %d", end_ts)
            logger.info("Migration summary: processed %d logs", total_processed)

    except Exception as e:
        logger.exception("Migration failed: %s", str(e))


if __name__ == "__main__":
    main()
