import logging
import os
from functools import partial
from typing import Any, Dict, Set

import polars as pl
import torch
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.common.data.collate import create_rl_batch
from src.common.datasets.recommendation_dataset import RecommendationDataset
from src.common.models.agent import DDPGAgent
from src.common.storage.s3_handler import S3Handler
from src.common.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)

BUCKET_NAME = os.getenv("BUCKET_NAME", "")
AWS_REGION = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

MONGODB_URL = os.getenv("MONGODB_URL", "")

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


def get_s3_handler(config: Dict[str, Any]) -> S3Handler:
    if config["use_minio"]:
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


def get_consented_uuids() -> Set[str]:
    try:
        client: MongoClient = MongoClient(MONGODB_URL)
        db = client.get_database()
        uuid_collection = db.Uuid

        consented_uuids = set()
        cursor = uuid_collection.find({"trainingConsent": True}, {"uuid": 1})

        for doc in cursor:
            consented_uuids.add(doc["uuid"])

        client.close()

        logger.info("Retrieved %d consented UUIDs from MongoDB", len(consented_uuids))
        return consented_uuids

    except Exception as e:
        logger.error("Error fetching consented UUIDs from MongoDB: %s", str(e))
        return set()


def main() -> None:
    try:
        s3_handler: S3Handler = get_s3_handler(config_dict)

        consented_uuids = get_consented_uuids()
        if not consented_uuids:
            logger.warning("No consented UUIDs found.")

        poster_viewed_df = s3_handler.get_all_activity_logs(
            log_name="poster_viewed",
            columns=["uuid", "session_id", "item_id", "watch_time", "created_at"],
        )

        assert poster_viewed_df is not None, "trainning data is None"

        poster_viewed_df = poster_viewed_df.filter(
            pl.col("uuid").is_in(consented_uuids)
        )
        logger.info(
            "Filtered data size after consent check: %d rows", poster_viewed_df.height
        )

        unique_item_ids = [
            id
            for id in poster_viewed_df.select("item_id").unique().to_series().to_list()
        ]

        item_embeddings = {}
        embedding_batch_size = config_dict["data"]["embedding_batch_size"]
        total_batches = (
            len(unique_item_ids) + embedding_batch_size - 1
        ) // embedding_batch_size

        logger.info(
            "Processing %d items in %d batches...", len(unique_item_ids), total_batches
        )
        total_batches = 1  # todo: remove this

        for i in range(total_batches):
            start_idx = i * embedding_batch_size
            end_idx = min((i + 1) * embedding_batch_size, len(unique_item_ids))
            batch = unique_item_ids[start_idx:end_idx]

            response = index.fetch(ids=list(batch), namespace="movies")

            vectors_dict = {
                id: vector.values for id, vector in response.vectors.items()
            }
            item_embeddings.update(vectors_dict)

            logger.info(
                "Batch %d/%d completed. Total embeddings: %d",
                i + 1,
                total_batches,
                len(item_embeddings),
            )

        valid_items = set(item_embeddings.keys())
        filtered_df = poster_viewed_df.filter(pl.col("item_id").is_in(valid_items))

        sorted_df = filtered_df.sort(["session_id", "created_at"])

        grouped_df = sorted_df.group_by("session_id").agg(
            [
                pl.col("item_id").alias("items"),
                pl.col("watch_time").alias("watch_times"),
            ]
        )

        grouped_df = grouped_df.filter(
            pl.col("items").list.len() >= config_dict["data"]["min_items"] + 1
        )

        train_df, test_df = train_test_split(
            grouped_df,
            test_size=config_dict["data"]["eval_ratio"],
            random_state=config_dict["random_seed"],
        )

        tensor_embeddings = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in item_embeddings.items()
        }

        train_dataset = RecommendationDataset(train_df, tensor_embeddings)
        test_dataset = RecommendationDataset(test_df, tensor_embeddings)

        collate_fn = partial(
            create_rl_batch,
            embedding_dim=config_dict["data"]["embedding_dim"],
            max_sequence_length=config_dict["data"]["max_sequence_length"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_dict["train"]["batch_size"],
            shuffle=config_dict["train"]["shuffle_flag"],
            num_workers=config_dict["train"]["num_workers"],
            collate_fn=collate_fn,
        )

        eval_loader = DataLoader(
            test_dataset,
            batch_size=config_dict["eval"]["batch_size"],
            shuffle=config_dict["eval"]["shuffle_flag"],
            num_workers=config_dict["eval"]["num_workers"],
            collate_fn=collate_fn,
        )

        agent = DDPGAgent(config_dict, train_loader, eval_loader)
        agent.train(num_epochs=config_dict["train"]["epochs"])

    except Exception as e:
        logger.exception("Error during ddpg model training: %s", str(e))


if __name__ == "__main__":
    main()
