import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import polars as pl
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import AutoReconnect, ConnectionFailure
from pymongo.server_api import ServerApi
from upstash_redis import Redis

from src.data.handlers.s3_handler import S3Handler
from src.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)


BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
AWS_REGION: str = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "")

MONGODB_URL: str = os.getenv("MONGODB_URL", "")
MONGODB_NAME: str = os.getenv("MONGODB_NAME", "")

TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")

REDIS_URL: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))
timestamp = datetime.now()
redis_client = Redis(url=REDIS_URL, token=REDIS_TOKEN)


class Config:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.movie_path: str = config["storage"]["movie"]
        self.video_path: str = config["storage"]["video"]
        self.image_path: str = config["storage"]["image"]
        self.trendings_path: str = config["storage"]["trendings"]
        self.use_minio: bool = config["use_minio"]
        self.sync_images: bool = config["sync_images"]
        self.collections: Dict[str, str] = config["mongodb"]["collections"]
        self.trending_prefix: str = config["redis"]["trending_prefix"]


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


def retry_operation(operation, max_retries=3, delay=2):
    """接続エラー時に処理を再試行するラッパー関数"""
    retries = 0
    while retries < max_retries:
        try:
            return operation()
        except (ConnectionFailure, AutoReconnect) as e:
            retries += 1
            if retries >= max_retries:
                raise
            logger.warning(
                f"MongoDB operation failed: {e}. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})"
            )
            time.sleep(delay)
            delay *= 2  # 指数バックオフ


def sync_movies(
    collection: Collection, movies_list: list[dict], timestamp: datetime
) -> None:
    if not movies_list:
        return

    # バッチサイズを小さくして処理
    batch_size = 100
    for i in range(0, len(movies_list), batch_size):
        batch = movies_list[i : i + batch_size]
        bulk_ops = []
        for movie in batch:
            movie["createdAt"] = timestamp
            movie["updatedAt"] = timestamp
            bulk_ops.append(
                UpdateOne({"movieId": movie["movieId"]}, {"$set": movie}, upsert=True)
            )

        if bulk_ops:

            def bulk_write_operation():
                result = collection.bulk_write(bulk_ops)
                logger.info(
                    f"Movies batch {i // batch_size + 1} upserted: {result.bulk_api_result}"
                )
                return result

            retry_operation(bulk_write_operation)


def sync_videos(
    collection: Collection, videos_list: list[dict], timestamp: datetime
) -> None:
    if not videos_list:
        return

    # バッチサイズを小さくして処理
    batch_size = 100
    for i in range(0, len(videos_list), batch_size):
        batch = videos_list[i : i + batch_size]
        bulk_ops = []
        for video in batch:
            if "videoId" not in video:
                continue

            video_data = {k: v for k, v in video.items() if k != "movieId"}
            bulk_ops.append(
                UpdateOne(
                    {"movieId": video["movieId"]},
                    {
                        "$set": {
                            "movieId": video["movieId"],
                            "createdAt": timestamp,
                            "updatedAt": timestamp,
                        },
                        "$push": {"videos": video_data},
                    },
                    upsert=True,
                )
            )

        if bulk_ops:

            def bulk_write_operation():
                result = collection.bulk_write(bulk_ops)
                logger.info(
                    f"Videos batch {i // batch_size + 1} upserted: {result.bulk_api_result}"
                )
                return result

            retry_operation(bulk_write_operation)


def sync_images(
    collection: Collection, images_list: list[dict], timestamp: datetime
) -> None:
    if not images_list:
        return

    # バッチサイズを小さくして処理
    batch_size = 100
    for i in range(0, len(images_list), batch_size):
        batch = images_list[i : i + batch_size]
        bulk_ops = []
        for image in batch:
            if "filePath" not in image:
                continue
            image_type = image.get("imageType")
            if not image_type:
                continue

            image_data = {
                k: v for k, v in image.items() if k not in ["movieId", "imageType"]
            }
            bulk_ops.append(
                UpdateOne(
                    {"movieId": image["movieId"]},
                    {
                        "$set": {
                            "movieId": image["movieId"],
                            "createdAt": timestamp,
                            "updatedAt": timestamp,
                        },
                        "$push": {image_type: image_data},
                    },
                    upsert=True,
                )
            )

        if bulk_ops:

            def bulk_write_operation():
                result = collection.bulk_write(bulk_ops)
                logger.info(
                    f"Images batch {i // batch_size + 1} upserted: {result.bulk_api_result}"
                )
                return result

            retry_operation(bulk_write_operation)


def sync_trendings(
    redis_client: Redis,
    trending_prefix: str,
    trendings_list: list[dict],
    timestamp: datetime,
) -> None:
    ONE_WEEK_SECONDS: int = 7 * 24 * 60 * 60

    if trendings_list:
        latest_trending: Dict[str, Any] = sorted(
            trendings_list, key=lambda x: x["created_at"]
        )[-1]
        trending_data: Dict[str, Any] = {
            "movieIds": latest_trending["movieIds"],
            "createdAt": timestamp.isoformat(),
            "updatedAt": timestamp.isoformat(),
        }
        redis_result = redis_client.set(
            trending_prefix, json.dumps(trending_data), ex=ONE_WEEK_SECONDS
        )
        logger.info("Trendings saved to Redis: %s", redis_result)


def main() -> None:
    try:
        config: Config = Config(config_dict)
        s3_handler: S3Handler = get_s3_handler(config)

        movies_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(config.movie_path)
        videos_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(config.video_path)
        images_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(config.image_path)
        trendings_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.trendings_path
        )

        mongo_movies_df: Optional[pl.DataFrame] = None
        mongo_videos_df: Optional[pl.DataFrame] = None
        mongo_images_df: Optional[pl.DataFrame] = None
        mongo_trendings_df: Optional[pl.DataFrame] = None

        if movies_df is not None:
            mongo_movies_df = movies_df.filter(pl.col("synced_at").is_null()).drop(
                ["synced_at", "created_at", "updated_at", "embedded_at"]
            )
        if videos_df is not None:
            mongo_videos_df = videos_df.filter(pl.col("synced_at").is_null()).drop(
                ["synced_at", "created_at", "updated_at", "embedded_at"]
            )
        if images_df is not None:
            mongo_images_df = images_df.filter(pl.col("synced_at").is_null()).drop(
                ["synced_at", "created_at", "updated_at", "embedded_at"]
            )
        if trendings_df is not None:
            mongo_trendings_df = trendings_df.filter(pl.col("synced_at").is_null())

        movies_list: List[Dict[str, Any]] = (
            mongo_movies_df.to_dicts() if mongo_movies_df is not None else []
        )
        videos_list: List[Dict[str, Any]] = (
            mongo_videos_df.to_dicts() if mongo_videos_df is not None else []
        )
        images_list: List[Dict[str, Any]] = (
            mongo_images_df.to_dicts() if mongo_images_df is not None else []
        )
        trendings_list: List[Dict[str, Any]] = (
            mongo_trendings_df.to_dicts() if mongo_trendings_df is not None else []
        )

        # タイムアウト設定を追加したMongoDBクライアント
        client: MongoClient = MongoClient(
            MONGODB_URL,
            server_api=ServerApi("1"),
            socketTimeoutMS=60000,  # ソケットタイムアウトを60秒に設定
            connectTimeoutMS=30000,  # 接続タイムアウトを30秒に設定
            serverSelectionTimeoutMS=30000,  # サーバー選択タイムアウトを30秒に設定
        )

        # 接続テスト
        def test_connection():
            client.admin.command("ping")
            return True

        if retry_operation(test_connection):
            logger.info("Successfully connected to MongoDB!")

        # データベース選択
        db = client[MONGODB_NAME]
        movies_collection: Collection = db[config.collections["movie"]]
        videos_collection: Collection = db[config.collections["videos"]]
        images_collection: Collection = db[config.collections["images"]]

        # 同期処理
        sync_movies(movies_collection, movies_list, timestamp)
        sync_videos(videos_collection, videos_list, timestamp)
        if config.sync_images:
            sync_images(images_collection, images_list, timestamp)
        sync_trendings(redis_client, config.trending_prefix, trendings_list, timestamp)

        # データフレーム更新
        if movies_df is not None:
            movies_df = movies_df.with_columns(
                pl.when(pl.col("synced_at").is_null())
                .then(pl.lit(timestamp))
                .otherwise(pl.col("synced_at"))
                .alias("synced_at")
            )
        if videos_df is not None:
            videos_df = videos_df.with_columns(
                pl.when(pl.col("synced_at").is_null())
                .then(pl.lit(timestamp))
                .otherwise(pl.col("synced_at"))
                .alias("synced_at")
            )
        if config.sync_images and images_df is not None:
            images_df = images_df.with_columns(
                pl.when(pl.col("synced_at").is_null())
                .then(pl.lit(timestamp))
                .otherwise(pl.col("synced_at"))
                .alias("synced_at")
            )
        if trendings_df is not None:
            trendings_df = trendings_df.with_columns(
                pl.when(pl.col("synced_at").is_null())
                .then(pl.lit(timestamp))
                .otherwise(pl.col("synced_at"))
                .alias("synced_at")
            )

        # S3/MinIOへのデータ保存
        if movies_df is not None:
            s3_handler.put_parquet_object(movies_df, config.movie_path)
        if videos_df is not None:
            s3_handler.put_parquet_object(videos_df, config.video_path)
        if config.sync_images and images_df is not None:
            s3_handler.put_parquet_object(images_df, config.image_path)
        if trendings_df is not None:
            s3_handler.put_parquet_object(trendings_df, config.trendings_path)

    except Exception as e:
        logger.exception("Error during TMDB data synchronization: %s", str(e))


if __name__ == "__main__":
    main()
