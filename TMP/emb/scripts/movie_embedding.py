import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytz
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

from src.data.handlers.movie_embedding_handler import MovieEmbeddingHandler
from src.data.handlers.s3_handler import S3Handler
from src.models.movie import Movie
from src.models.schemas import (
    failures_schema,
    movies_schema,
    validate_schema,
)
from src.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)

BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
AWS_REGION: str = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))
timestamp = datetime.now()


class Config:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.movie_path: str = config["storage"]["movie"]
        self.failures_path: str = config["storage"]["failures"]
        self.use_minio: bool = config["use_minio"]
        self.batch_size: int = config["pinecone"]["batch_size"]
        self.text_model_name: str = config["model"]["text"]["model_name"]
        self.image_model_name: str = config["model"]["image"]["model_name"]
        self.text_vector_dim: int = config["model"]["text"]["vector_dim"]
        self.image_vector_dim: int = config["model"]["image"]["vector_dim"]


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


def process_movie(
    handler: MovieEmbeddingHandler, movie_dict: Dict
) -> Optional[np.ndarray]:
    try:
        movie = Movie(**movie_dict)
        embedding = handler.store_movie(movie)
        logger.debug("Successfully processed movie: %s", movie.movieId)
        return embedding
    except Exception as e:
        logger.error(
            "Error processing movie %s: %s",
            movie_dict.get("movieId", "unknown"),
            str(e),
        )
        return None


def main() -> None:
    try:
        config: Config = Config(config_dict)
        s3_handler: S3Handler = get_s3_handler(config)

        embedding_handler = MovieEmbeddingHandler(
            text_model_name=config.text_model_name,
            image_model_name=config.image_model_name,
            text_vector_dim=config.text_vector_dim,
            image_vector_dim=config.image_vector_dim,
        )

        existing_movies_df = s3_handler.get_dataframe(config.movie_path)
        existing_failures_df = s3_handler.get_dataframe(config.failures_path)

        if existing_movies_df is None:
            logger.error("Failed to load movies dataframe from: %s", config.movie_path)
            return

        existing_failed_ids = []
        if existing_failures_df is not None:
            existing_failed_ids = existing_failures_df.filter(
                pl.col("type") == "MOVIE_EMBEDDING_ERROR"
            )["movieId"].to_list()

        pending_movies = existing_movies_df.filter(
            (pl.col("embedded_at").is_null())
            # & (pl.col("synced_at").is_not_null())
            & (~pl.col("movieId").is_in(existing_failed_ids))
        )

        pending_movie_dicts = pending_movies.to_dicts()
        if not pending_movie_dicts:
            logger.warning("No pending movies found at path: %s", config.movie_path)
            return

        batch_vectors = []
        success_ids = []
        new_failures = []

        for movie_dict in pending_movie_dicts:
            embedding = process_movie(embedding_handler, movie_dict)
            movie_id = movie_dict.get("movieId")
            if embedding is not None:
                success_ids.append(movie_id)
                batch_vectors.append(
                    {
                        "id": str(movie_id),
                        "values": embedding,
                        "metadata": {"movieId": movie_id},
                    }
                )
                if len(batch_vectors) >= config.batch_size:
                    index.upsert(vectors=batch_vectors, namespace="movies")
                    logger.info(
                        "Upserted a batch of %d vectors to Pinecone", len(batch_vectors)
                    )
                    batch_vectors = []
            else:
                new_failures.append(
                    {
                        "movieId": movie_id if movie_id is not None else "unknown",
                        "type": "MOVIE_EMBEDDING_ERROR",
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                )

        if batch_vectors:
            index.upsert(vectors=batch_vectors, namespace="movies")
            logger.info(
                "Upserted final batch of %d vectors to Pinecone", len(batch_vectors)
            )

        if success_ids:
            updated_movies_df = existing_movies_df.with_columns(
                pl.when(pl.col("movieId").is_in(success_ids))
                .then(pl.lit(timestamp))
                .otherwise(pl.col("embedded_at"))
                .alias("embedded_at")
            )
            is_movies_valid, movies_errors = validate_schema(
                updated_movies_df, movies_schema
            )
            if not is_movies_valid:
                logger.error("Movies schema validation failed: %s", movies_errors)
            else:
                s3_handler.put_parquet_object(updated_movies_df, config.movie_path)
                logger.info("Updated embedded_at for %d movies", len(success_ids))
        else:
            logger.info("No movies processed successfully.")

        if new_failures:
            new_failures_df = pl.DataFrame(new_failures)
            if existing_failures_df is not None and not existing_failures_df.is_empty():
                updated_failures_df = pl.concat([existing_failures_df, new_failures_df])
            else:
                updated_failures_df = new_failures_df
            is_failures_valid, failures_errors = validate_schema(
                updated_failures_df, failures_schema
            )
            if not is_failures_valid:
                logger.error("Failures schema validation failed: %s", failures_errors)
            else:
                s3_handler.put_parquet_object(updated_failures_df, config.failures_path)
                logger.warning(
                    "Recorded %d movie processing failures", len(new_failures)
                )

    except Exception as e:
        logger.exception("Error during movie data embedding: %s", str(e))


if __name__ == "__main__":
    main()
