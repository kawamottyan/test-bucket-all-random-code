from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, cast

import polars as pl

from src.data.handlers.s3_handler import S3Handler
from src.models.log import (
    DetailViewedLog,
    InteractionLog,
    InteractionType,
    PlayStartedLog,
    PosterViewedLog,
)
from src.models.schemas import (
    SchemaType,
    log_schemas,
    validate_schema,
)
from src.utils.general import setup_logger

logger = setup_logger(__name__)


class RedisLogHandler:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def get_last_processed_timestamp(self, migration_marker: str) -> int:
        last_ts = self.redis_client.get(migration_marker)
        return int(last_ts) if last_ts else 0

    def update_last_processed_timestamp(
        self, migration_marker: str, timestamp: int
    ) -> None:
        self.redis_client.set(migration_marker, timestamp)

    def get_interaction_keys(
        self,
        interaction_prefix: str,
        batch_size: int,
        start_timestamp: int,
        end_timestamp: int,
    ) -> List[str]:
        keys = []
        cursor = 0

        while True:
            cursor, batch = self.redis_client.scan(
                cursor, f"{interaction_prefix}*", batch_size
            )

            for key in batch:
                try:
                    parts = key.split(":")
                    if len(parts) >= 3:
                        key_timestamp = int(parts[2])
                        if start_timestamp <= key_timestamp < end_timestamp:
                            keys.append(key)
                except (IndexError, ValueError):
                    logger.warning("Invalid key format: %s", key)

            if cursor == 0:
                break

        return keys

    def fetch_interaction_logs(self, keys: List[str]) -> List[Dict[str, Any]]:
        logs = []

        for key in keys:
            try:
                log_data = self.redis_client.hgetall(key)
                if log_data:
                    parts = key.split(":")
                    if len(parts) >= 3:
                        session_id = parts[1]

                        log_data["session_id"] = session_id
                        logs.append(log_data)
            except Exception as e:
                logger.error("Error fetching log for key %s: %s", key, str(e))

        return logs

    def delete_migrated_logs(self, keys: List[str]) -> int:
        if not keys:
            return 0

        try:
            deleted_count = self.redis_client.delete(*keys)
            logger.info("Deleted %d logs from Redis", deleted_count)
            return deleted_count
        except Exception as e:
            logger.error("Failed to delete logs from Redis: %s", str(e))
            return 0


def save_and_delete_interaction_logs(
    self,
    s3_handler: S3Handler,
    logs: List[Dict[str, Any]],
    keys: List[str],
    timestamp: datetime,
) -> Dict[str, int]:
    if not logs:
        logger.info("No logs to save")
        return {}

    logs_by_type = defaultdict(list)
    keys_by_type = defaultdict(list)
    valid_keys = []
    invalid_keys = []

    for i, log in enumerate(logs):
        try:
            created_at_str = log.get("created_at")
            if not created_at_str:
                logger.warning("Missing created_at in log: %s", log)
                invalid_keys.append(keys[i])
                continue

            try:
                if not isinstance(created_at_str, datetime):
                    datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except (ValueError, TypeError, AttributeError):
                logger.warning("Invalid created_at format in log: %s", created_at_str)
                invalid_keys.append(keys[i])
                continue

            log_with_ts = {**log, "migrated_at": timestamp}

            interaction_type = log_with_ts.get("interaction_type")
            if not interaction_type:
                logger.warning("Missing interaction_type in log: %s", log)
                invalid_keys.append(keys[i])
                continue

            try:
                interaction_type = InteractionType(interaction_type)
            except ValueError:
                logger.warning("Invalid interaction_type: %s", interaction_type)
                invalid_keys.append(keys[i])
                continue

            if interaction_type == InteractionType.POSTER_VIEWED:
                parsed_log = PosterViewedLog(**log_with_ts).model_dump()
            elif interaction_type == InteractionType.DETAIL_VIEWED:
                parsed_log = DetailViewedLog(**log_with_ts).model_dump()
            elif interaction_type == InteractionType.PLAY_STARTED:
                parsed_log = PlayStartedLog(**log_with_ts).model_dump()
            else:
                parsed_log = InteractionLog(**log_with_ts).model_dump()

            logs_by_type[interaction_type].append(parsed_log)
            keys_by_type[interaction_type].append(keys[i])
            valid_keys.append(keys[i])

        except Exception as e:
            logger.warning("Invalid log: %s", e)
            invalid_keys.append(keys[i])

    results = {}

    date_path = (
        f"year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}"
    )
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    for log_type, type_logs in logs_by_type.items():
        if not type_logs:
            continue

        try:
            df = pl.DataFrame(type_logs)

            try:
                if "created_at" in df.columns:
                    df = df.with_columns(pl.col("created_at").cast(pl.Datetime))
            except Exception as e:
                logger.error("Failed to convert created_at to datetime: %s", str(e))
                results[str(log_type)] = 0
                continue

            schema = log_schemas.get(log_type)
            if schema:
                typed_schema = cast(SchemaType, schema)
                is_valid, errors = validate_schema(df, typed_schema)
                if not is_valid:
                    logger.error(
                        "Schema validation failed for %s: %s", log_type, errors
                    )
                    results[str(log_type)] = 0
                    continue

            type_value = log_type.value if hasattr(log_type, "value") else str(log_type)
            s3_key = f"log_storage/{type_value}/{date_path}/{timestamp_str}.parquet"

            s3_handler.upload_parquet(df, s3_key)

            type_keys = keys_by_type[log_type]
            deleted_count = self.delete_migrated_logs(type_keys)

            results[type_value] = len(type_logs)
            logger.info(
                "Saved %d %s logs to %s and deleted %d logs from Redis",
                len(type_logs),
                type_value,
                s3_key,
                deleted_count,
            )

        except Exception as e:
            logger.error("Failed to save %s logs: %s", str(log_type), str(e))
            results[str(log_type)] = 0

        logger.info("Deleting %d invalid logs from Redis", len(invalid_keys))
        self.delete_migrated_logs(invalid_keys)

    return results
