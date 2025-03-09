import gzip
import io
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import polars as pl
import requests
from dotenv import load_dotenv

from src.data.handlers.s3_handler import S3Handler
from src.data.handlers.tmdb_handler import fetch_movie_details, fetch_trending_ids
from src.models.schemas import (
    failures_schema,
    images_schema,
    movies_schema,
    trendings_schema,
    validate_schema,
    videos_schema,
)
from src.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)


BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
AWS_REGION: str = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "")

TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))
timestamp = datetime.now()


class Config:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.movie_path: str = config["storage"]["movie"]
        self.video_path: str = config["storage"]["video"]
        self.image_path: str = config["storage"]["image"]
        self.failures_path: str = config["storage"]["failures"]
        self.trendings_path: str = config["storage"]["trendings"]
        self.use_minio: bool = config["use_minio"]
        self.fetch_limit: int = config["movie"]["fetch_limit"]


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


def fetch_tmdb_ids(date: str) -> list:
    url = f"http://files.tmdb.org/p/exports/movie_ids_{date}.json.gz"
    try:
        response = requests.get(url)
        response.raise_for_status()

        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            tmdb_ids = []
            for line in f:
                data = json.loads(line.decode("utf-8"))
                tmdb_ids.append(data.get("id"))
        return tmdb_ids
    except Exception as e:
        logger.exception("Error fetching data for %s: %s", date, str(e))
        return []


def get_all_tmdb_ids(lookup_days: int) -> list:
    today = datetime.today()
    for i in range(0, lookup_days):
        date = (today - timedelta(days=i)).strftime("%m_%d_%Y")
        tmdb_ids = fetch_tmdb_ids(date)
        if tmdb_ids:
            logger.info("Found %s movie IDs for %s.", len(tmdb_ids), date)
            return tmdb_ids
    else:
        logger.error("No movie data found in the last %s days.", lookup_days)
        return []


def add_timestamps(data: dict, current_time: datetime) -> dict:
    data["created_at"] = current_time
    data["updated_at"] = current_time
    data["synced_at"] = None
    data["embedded_at"] = None
    return data


def remove_timezone_info(df: pl.DataFrame) -> pl.DataFrame:
    if df is None:
        return None

    for col_name, dtype in df.schema.items():
        if "datetime" in str(dtype).lower():
            if hasattr(dtype, "time_zone") and dtype.time_zone is not None:
                df = df.with_columns(
                    [pl.col(col_name).dt.replace_time_zone(None).alias(col_name)]
                )

    return df


def main() -> None:
    try:
        config: Config = Config(config_dict)
        s3_handler: S3Handler = get_s3_handler(config)

        existing_movies_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.movie_path
        )
        existing_videos_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.video_path
        )
        existing_images_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.image_path
        )
        existing_failures_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.failures_path
        )
        existing_trendings_df: Optional[pl.DataFrame] = s3_handler.get_dataframe(
            config.trendings_path
        )

        trending_tmdb_ids: list[int] = fetch_trending_ids(TMDB_API_KEY)

        if existing_movies_df is not None and not existing_movies_df.is_empty():
            existing_tmdb_ids = existing_movies_df["movieId"].to_list()
        else:
            existing_tmdb_ids = []

        if existing_failures_df is not None and not existing_failures_df.is_empty():
            existing_failed_tmdb_ids = existing_failures_df["movieId"].to_list()
        else:
            existing_failed_tmdb_ids = []

        all_tmdb_ids = get_all_tmdb_ids(lookup_days=3)
        new_tmdb_ids = list(
            set(all_tmdb_ids) - set(existing_tmdb_ids) - set(existing_failed_tmdb_ids)
        )

        batch_tmdb_ids = list(
            set(new_tmdb_ids[: config.fetch_limit] + trending_tmdb_ids)
        )

        movies_list = []
        videos_list = []
        images_list = []
        failures_list = []

        for tmdb_id in batch_tmdb_ids:
            movie, movie_videos, movie_images, failure = fetch_movie_details(
                TMDB_API_KEY, tmdb_id, timestamp
            )

            if failure:
                if "movieId" in failure and failure["movieId"] is not None:
                    failure["movieId"] = str(failure["movieId"])
                failures_list.append(failure)
                continue
            if movie:
                movie_data = add_timestamps(movie.model_dump(), timestamp)
                movies_list.append(movie_data)

            if movie_videos:
                for video in movie_videos.videos:
                    video_data = add_timestamps(video.model_dump(), timestamp)
                    video_data["movieId"] = movie_videos.movieId
                    videos_list.append(video_data)

            if movie_images:
                for category in ["backdrops", "logos", "posters"]:
                    for image in getattr(movie_images, category, []):
                        image_data = add_timestamps(image.model_dump(), timestamp)
                        image_data["movieId"] = movie_images.movieId
                        image_data["imageType"] = category
                        images_list.append(image_data)

        new_movies_df = pl.DataFrame(movies_list)
        new_videos_df = pl.DataFrame(videos_list)
        new_images_df = pl.DataFrame(images_list)
        new_failures_df = pl.DataFrame(failures_list)
        new_trendings_df = pl.DataFrame(
            {
                "movieIds": [trending_tmdb_ids],
                "created_at": [timestamp],
                "updated_at": [timestamp],
                "synced_at": [None],
            }
        )

        if movies_list:
            new_movies_df = pl.DataFrame(movies_list)
            new_movies_df = remove_timezone_info(new_movies_df)
            if existing_movies_df is not None and not existing_movies_df.is_empty():
                movies_df = pl.concat([existing_movies_df, new_movies_df])
            else:
                movies_df = new_movies_df
        else:
            movies_df = (
                existing_movies_df if existing_movies_df is not None else pl.DataFrame()
            )

        if videos_list:
            new_videos_df = pl.DataFrame(videos_list)
            new_videos_df = remove_timezone_info(new_videos_df)
            if existing_videos_df is not None and not existing_videos_df.is_empty():
                videos_df = pl.concat([existing_videos_df, new_videos_df])
            else:
                videos_df = new_videos_df
        else:
            videos_df = (
                existing_videos_df if existing_videos_df is not None else pl.DataFrame()
            )

        if images_list:
            new_images_df = pl.DataFrame(images_list)
            new_images_df = remove_timezone_info(new_images_df)
            if existing_images_df is not None and not existing_images_df.is_empty():
                images_df = pl.concat([existing_images_df, new_images_df])
            else:
                images_df = new_images_df
        else:
            images_df = (
                existing_images_df if existing_images_df is not None else pl.DataFrame()
            )

        if failures_list:
            new_failures_df = pl.DataFrame(failures_list)
            new_failures_df = remove_timezone_info(new_failures_df)
            if existing_failures_df is not None and not existing_failures_df.is_empty():
                failures_df = pl.concat([existing_failures_df, new_failures_df])
            else:
                failures_df = new_failures_df
        else:
            failures_df = (
                existing_failures_df
                if existing_failures_df is not None
                else pl.DataFrame()
            )

        if existing_trendings_df is not None and not existing_trendings_df.is_empty():
            trendings_df = pl.concat([existing_trendings_df, new_trendings_df])
        else:
            trendings_df = new_trendings_df

        is_movies_valid, movies_errors = validate_schema(movies_df, movies_schema)
        is_videos_valid, videos_errors = validate_schema(videos_df, videos_schema)
        is_images_valid, images_errors = validate_schema(images_df, images_schema)
        is_failures_valid, failures_errors = validate_schema(
            failures_df, failures_schema
        )
        is_trendings_valid, trendings_errors = validate_schema(
            trendings_df, trendings_schema
        )

        if not is_movies_valid:
            logger.error("Movies schema validation failed: %s", movies_errors)
        if not is_videos_valid:
            logger.error("Videos schema validation failed: %s", videos_errors)
        if not is_images_valid:
            logger.error("Images schema validation failed: %s", images_errors)
        if not is_failures_valid:
            logger.error("Failures schema validation failed: %s", failures_errors)
        if not is_trendings_valid:
            logger.error("Trendings schema validation failed: %s", trendings_errors)

        if is_movies_valid:
            s3_handler.put_parquet_object(movies_df, config.movie_path)
        if is_videos_valid:
            s3_handler.put_parquet_object(videos_df, config.video_path)
        if is_images_valid:
            s3_handler.put_parquet_object(images_df, config.image_path)
        if is_failures_valid:
            s3_handler.put_parquet_object(failures_df, config.failures_path)
        if is_trendings_valid:
            s3_handler.put_parquet_object(trendings_df, config.trendings_path)

    except Exception as e:
        logger.exception("Error during TMDB data fetch and processing: %s", str(e))


if __name__ == "__main__":
    main()
