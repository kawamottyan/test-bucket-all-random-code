import logging
from dotenv import load_dotenv
import os
from src.recommendation.common.utils.general import load_config, set_random_seed
from src.recommendation.common.storage.s3_handler import S3Handler
import polars as pl
from sklearn.model_selection import train_test_split
from src.recommendation.common.datasets.interaction_dataset import RecommendationDataset
import torch
from functools import partial
from src.recommendation.common.data.collate.reinforcement_collate import pad_collate
from torch.utils.data import DataLoader
from pinecone.grpc import PineconeGRPC as Pinecone
from src.recommendation.common.models.agent import DDPGAgent

logger = logging.getLogger(__name__)
load_dotenv()


USE_MINIO = os.getenv("USE_MINIO")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


def main() -> None:
    try:
        config = load_config("config.yaml")
        logging.basicConfig(level=config["log_level"].upper())
        set_random_seed(config["random_seed"])

        if config.get("use_minio", True):
            s3_handler = S3Handler(
                aws_region=AWS_REGION,
                bucket_name=BUCKET_NAME,
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ROOT_USER,
                aws_secret_access_key=MINIO_ROOT_PASSWORD,
            )
        else:
            s3_handler = S3Handler(aws_region=AWS_REGION, bucket_name=BUCKET_NAME)

        poster_viewed_df = s3_handler.get_all_activity_logs(
            log_name="poster_viewed",
            columns=["user_id", "item_id", "watch_time", "event_time"],
        )

        unique_item_ids = [
            id
            for id in poster_viewed_df.select("item_id").unique().to_series().to_list()
        ]

        item_embeddings = {}
        batch_size = 1000
        total_batches = (len(unique_item_ids) + batch_size - 1) // batch_size

        print(f"Processing {len(unique_item_ids)} items in {total_batches} batches...")
        total_batches = 5  # todo

        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unique_item_ids))
            batch = unique_item_ids[start_idx:end_idx]

            response = index.fetch(ids=list(batch), namespace="movies")

            vectors_dict = {
                id: vector.values for id, vector in response.vectors.items()
            }
            item_embeddings.update(vectors_dict)

            print(
                f"Batch {i + 1}/{total_batches} completed. Total embeddings: {len(item_embeddings)}"
            )

        valid_items = set(item_embeddings.keys())
        filtered_df = poster_viewed_df.filter(pl.col("item_id").is_in(valid_items))

        min_items = 5
        grouped_df = filtered_df.group_by("user_id").agg(
            [
                pl.col("item_id").alias("items"),
                pl.col("watch_time").alias("watch_times"),
            ]
        )

        grouped_df = grouped_df.filter(pl.col("items").list.len() >= min_items + 1)

        train_df, test_df = train_test_split(
            grouped_df,
            test_size=config["test_ratio"],
            random_state=config["random_seed"],
        )

        tensor_embeddings = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in item_embeddings.items()
        }

        train_dataset = RecommendationDataset(train_df, tensor_embeddings)
        test_dataset = RecommendationDataset(test_df, tensor_embeddings)

        custom_collate_fn = partial(
            pad_collate,
            frame_size=config["frame_size"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=config["train_shuffle_flag"],
            num_workers=config["train_num_workers"],
            collate_fn=custom_collate_fn,
        )

        eval_loader = DataLoader(
            test_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=config["eval_shuffle_flag"],
            num_workers=config["eval_num_workers"],
            collate_fn=custom_collate_fn,
        )

        agent = DDPGAgent(config, train_loader, eval_loader)
        agent.train(num_epochs=config["train_epochs"])

    except Exception as e:
        logger.exception("Error during ddpg model training: %s", str(e))


if __name__ == "__main__":
    main()
