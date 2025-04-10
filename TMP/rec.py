import marimo

__generated_with = "0.12.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import random
    import sys
    from pathlib import Path
    import pickle
    from io import BytesIO
    from typing import Any, Dict, Set, Optional, Union, TypedDict,List
    import os

    import numpy as np
    from sklearn.model_selection import train_test_split
    import polars as pl
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    import yaml
    import logging
    from dotenv import load_dotenv
    from functools import partial
    import boto3
    import botocore
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pymongo import MongoClient
    return (
        Any,
        BytesIO,
        DataLoader,
        Dataset,
        Dict,
        F,
        List,
        MongoClient,
        Optional,
        Path,
        Pinecone,
        Set,
        TypedDict,
        Union,
        boto3,
        botocore,
        load_dotenv,
        logging,
        nn,
        np,
        os,
        partial,
        pickle,
        pl,
        random,
        sys,
        torch,
        train_test_split,
        yaml,
    )


@app.cell
def _(logging, sys):
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False

        return logger
    return (setup_logger,)


@app.cell
def _(yaml):
    def load_config(config_path: str) -> dict:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ValueError(f"Config file {config_path} is empty")

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")
    return (load_config,)


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell
def _(os):
    BUCKET_NAME = os.getenv("BUCKET_NAME", "")
    AWS_REGION = os.getenv("AWS_REGION", "")
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
    MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "")
    MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_HOST = os.getenv("PINECONE_HOST", "")

    MONGODB_URL = os.getenv("MONGODB_URL", "")
    return (
        AWS_REGION,
        BUCKET_NAME,
        MINIO_ENDPOINT,
        MINIO_ROOT_PASSWORD,
        MINIO_ROOT_USER,
        MONGODB_URL,
        PINECONE_API_KEY,
        PINECONE_HOST,
    )


@app.cell
def _(Any, Dict, load_config, logging, setup_logger):
    logger = setup_logger(__name__)
    config_dict: Dict[str, Any] = load_config("config.yaml")
    log_level: str = config_dict.get("logging", {}).get("level", "INFO")
    logger.setLevel(getattr(logging, log_level))
    return config_dict, log_level, logger


@app.cell
def _(PINECONE_API_KEY, PINECONE_HOST, Pinecone):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    return index, pc


@app.cell
def _(
    Any,
    BytesIO,
    Dict,
    Optional,
    Path,
    Union,
    boto3,
    botocore,
    logger,
    pickle,
    pl,
    torch,
):
    class FileFormat:
        PARQUET = "parquet"
        CSV = "csv"
        PICKLE = "pickle"
        PT = "pt"


    class S3Handler:
        def __init__(
            self,
            aws_region: str,
            bucket_name: str,
            endpoint_url=None,
            verify_ssl=False,
            aws_access_key_id=None,
            aws_secret_access_key=None,
        ):
            client_kwargs: Dict[str, Union[str, bool, None, boto3.session.Config]] = {
                "region_name": aws_region,
            }

            if endpoint_url:
                client_kwargs.update(
                    {
                        "endpoint_url": endpoint_url,
                        "verify": verify_ssl,
                        "aws_access_key_id": aws_access_key_id,
                        "aws_secret_access_key": aws_secret_access_key,
                    }
                )
                client_kwargs["config"] = boto3.session.Config(signature_version="s3v4")

            self.s3_client = boto3.client("s3", **client_kwargs)
            self.bucket_name = bucket_name

        def _bucket_exists(self) -> bool:
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                return True
            except botocore.exceptions.ClientError as e:
                error_code = int(e.response["Error"]["Code"])
                if error_code == 404:
                    return False
                raise

        def _verify_bucket(self) -> None:
            try:
                if not self._bucket_exists():
                    logger.error("Bucket does not exist: %s", self.bucket_name)
                    raise ValueError(f"Bucket {self.bucket_name} does not exist")
                else:
                    logger.info("Bucket exists: %s", self.bucket_name)

            except botocore.exceptions.ClientError as e:
                logger.error("Failed to verify bucket %s: %s", self.bucket_name, e)
                raise

        def put_parquet_object(self, df: pl.DataFrame, s3_key: str) -> None:
            parquet_buffer = BytesIO()
            df.write_parquet(parquet_buffer)
            parquet_buffer.seek(0)

            logger.info("Uploading Parquet data to s3://%s/%s", self.bucket_name, s3_key)
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=s3_key, Body=parquet_buffer
            )
            logger.info("Uploaded Parquet data to s3://%s/%s", self.bucket_name, s3_key)

        def _get_object(self, s3_key: str):
            try:
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                return obj
            except self.s3_client.exceptions.NoSuchKey:
                logger.info("No object found at s3://%s/%s", self.bucket_name, s3_key)
                return None

        def _get_file_format(self, s3_key: str) -> str:
            extension = Path(s3_key).suffix.lower()
            format_map = {
                ".parquet": FileFormat.PARQUET,
                ".csv": FileFormat.CSV,
                ".pickle": FileFormat.PICKLE,
                ".pkl": FileFormat.PICKLE,
                ".pt": FileFormat.PT,
            }
            if extension in format_map:
                return format_map[extension]
            raise ValueError(f"Unsupported file extension: {extension}")

        def get_dataframe(
            self,
            s3_key: str,
            columns: Optional[list[str]] = None,
        ) -> Optional[pl.DataFrame]:
            try:
                logger.info("Fetching dataframe from s3://%s/%s", self.bucket_name, s3_key)
                obj = self._get_object(s3_key)
                if not obj:
                    return None

                file_format = self._get_file_format(s3_key)

                with BytesIO(obj["Body"].read()) as file_body:
                    if file_format == FileFormat.PARQUET:
                        df = pl.read_parquet(file_body)
                    elif file_format == FileFormat.CSV:
                        df = pl.read_csv(file_body)
                    else:
                        logger.error(
                            "Unsupported file format for dataframe: %s", file_format
                        )
                        return None

                    if columns:
                        df = df.select(columns)
                    return df

            except Exception as e:
                logger.error("Error retrieving dataframe from S3: %s", e)
                return None

        def get_pickle_object(
            self,
            s3_key: str,
        ) -> Optional[Any]:
            try:
                logger.info(
                    "Fetching pickle object from s3://%s/%s", self.bucket_name, s3_key
                )
                obj = self._get_object(s3_key)
                if not obj:
                    return None

                with BytesIO(obj["Body"].read()) as file_body:
                    return pickle.load(file_body)

            except Exception as e:
                logger.error("Error retrieving pickle object from S3: %s", e)
                return None

        def get_torch_model(self, s3_key: str) -> Optional[torch.jit.ScriptModule]:
            try:
                logger.info(
                    "Fetching torch model from s3://%s/%s", self.bucket_name, s3_key
                )
                obj = self._get_object(s3_key)
                if not obj:
                    return None

                with BytesIO(obj["Body"].read()) as file_body:
                    return torch.jit.load(file_body)

            except Exception as e:
                logger.error("Error retrieving torch model from S3: %s", e)
                return None

        def get_all_logs(
            self,
            log_name: str,
            columns: Optional[list[str]] = None,
        ) -> Optional[pl.DataFrame]:
            try:
                prefix = f"log_storage/{log_name.lower()}/"
                paginator = self.s3_client.get_paginator("list_objects_v2")
                dfs = []

                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            if obj["Key"].endswith(".parquet"):
                                df = self.get_dataframe(obj["Key"], columns)
                                if df is not None and not df.is_empty():
                                    dfs.append(df)

                return pl.concat(dfs) if dfs else None

            except Exception as e:
                logger.error("Error merging activity logs: %s", e)
                return None

        def upload_parquet(self, df: pl.DataFrame, s3_key: str) -> None:
            parquet_buffer = BytesIO()
            df.write_parquet(parquet_buffer)
            parquet_buffer.seek(0)

            logger.info("Uploading Parquet data to s3://%s/%s", self.bucket_name, s3_key)
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=s3_key, Body=parquet_buffer
            )
            logger.info("Uploaded Parquet data to s3://%s/%s", self.bucket_name, s3_key)
    return FileFormat, S3Handler


@app.cell
def _(AWS_REGION, Any, BUCKET_NAME, Dict, S3Handler):
    def get_s3_handler(config: Dict[str, Any]) -> S3Handler:
        return S3Handler(
            aws_region=AWS_REGION,
            bucket_name=BUCKET_NAME,
        )
    return (get_s3_handler,)


@app.cell
def _(S3Handler, config_dict, get_s3_handler):
    s3_handler: S3Handler = get_s3_handler(config_dict)
    return (s3_handler,)


@app.cell
def _(MONGODB_URL, MongoClient, Set, logger):
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
    return (get_consented_uuids,)


@app.cell
def _(get_consented_uuids):
    consented_uuids = get_consented_uuids()
    return (consented_uuids,)


@app.cell
def _(s3_handler):
    poster_viewed_df = s3_handler.get_all_logs(
        log_name="poster_viewed",
        columns=["uuid", "session_id", "movie_id", "watch_time", "created_at"],
    )
    poster_viewed_df
    return (poster_viewed_df,)


@app.cell
def _(s3_handler):
    recommendation_df = s3_handler.get_all_logs(
        log_name="recommendation",
        columns=["uuid", "session_id", "movie_ids", "created_at"],
    )
    recommendation_df
    return (recommendation_df,)


@app.cell
def _(pl, recommendation_df):
    # todo
    result_df = (
        recommendation_df
        .sort("created_at")
        .with_columns([
             pl.arange(0, pl.count()).over("session_id").alias("row_nr")
        ])
        .with_columns(
             pl.col("row_nr").max().over("session_id").alias("max_row_nr")
        )
        .filter((pl.col("row_nr") == 1) | (pl.col("row_nr") == pl.col("max_row_nr")))
        .drop(["row_nr", "max_row_nr"])
    )
    result_df
    return (result_df,)


@app.cell
def _(consented_uuids, pl, poster_viewed_df):
    train_df = poster_viewed_df.filter(pl.col("uuid").is_in(consented_uuids))
    return (train_df,)


@app.cell
def _(pl, train_df):
    unique_item_ids = (
        train_df.select(pl.col("movie_id").cast(pl.Utf8))
        .unique()
        .to_series()
        .to_list()
    )
    return (unique_item_ids,)


@app.cell
def _(config_dict, unique_item_ids):
    item_embeddings = {}
    embedding_batch_size = config_dict["data"]["embedding_batch_size"]
    total_batch_size = (
        len(unique_item_ids) + embedding_batch_size - 1
    ) // embedding_batch_size

    total_batch_size = 10  # todo: remove this
    return embedding_batch_size, item_embeddings, total_batch_size


@app.cell
def _(
    embedding_batch_size,
    index,
    item_embeddings,
    total_batch_size,
    unique_item_ids,
):
    for count in range(total_batch_size):
        start_idx = count * embedding_batch_size
        end_idx = min((count + 1) * embedding_batch_size, len(unique_item_ids))
        batch = unique_item_ids[start_idx:end_idx]

        response = index.fetch(ids=list(batch), namespace="movies")

        vectors_dict = {
            int(id): vector.values for id, vector in response.vectors.items()
        }
        item_embeddings.update(vectors_dict)
    return batch, count, end_idx, response, start_idx, vectors_dict


@app.cell
def _(item_embeddings, pl, train_df):
    valid_items = {int(item) for item in item_embeddings.keys()}
    filtered_df = train_df.filter(pl.col("movie_id").is_in(valid_items))

    sorted_df = filtered_df.sort(["session_id", "created_at"])

    grouped_df = sorted_df.group_by("session_id").agg(
        [
            pl.col("movie_id").alias("items"),
            pl.col("watch_time").alias("watch_times"),
            pl.col("created_at").alias("item_created_ats"),
            pl.len().alias("item_count")
        ]
    )

    filtered_grouped_df = grouped_df.filter(pl.col("item_count") >= 50)
    return filtered_df, filtered_grouped_df, grouped_df, sorted_df, valid_items


@app.cell
def _(filtered_grouped_df):
    filtered_grouped_df
    return


@app.cell
def _(pl, result_df):
    grouped_recommendation_df = result_df.group_by("session_id").agg(
        [
            pl.col("movie_ids").alias("action_lists"),
            pl.col("created_at").alias("rec_created_ats"),
            pl.len().alias("rec_count")
        ]
    )
    return (grouped_recommendation_df,)


@app.cell
def _(grouped_recommendation_df):
    grouped_recommendation_df
    return


@app.cell
def _(filtered_grouped_df, grouped_recommendation_df):
    joined_df = filtered_grouped_df.join(
        grouped_recommendation_df,
        on="session_id",
        how="inner"
    )
    joined_df
    return (joined_df,)


@app.cell
def generate_state_flags():
    def generate_state_flags(row):
        items = row["items"]
        item_count = row["item_count"]
        watch_times = row["watch_times"]
        item_timestamps = row["item_created_ats"]
        recommendation_timestamps = row["rec_created_ats"]

        state_done_flags = [False] * item_count

        for rec_time in recommendation_timestamps:
            last_item_before_rec = -1

            for i, item_time in enumerate(item_timestamps):
                if item_time < rec_time:
                    last_item_before_rec = i
                else:
                    break

            if last_item_before_rec >= 0:
                state_done_flags[last_item_before_rec] = True

        return state_done_flags
    return (generate_state_flags,)


@app.cell
def _(generate_state_flags, joined_df, pl):
    result = joined_df.with_columns([
        pl.struct(["items", "item_count", "watch_times", "item_created_ats", "rec_created_ats"])
        .map_elements(generate_state_flags, return_dtype=pl.List(pl.Boolean))
        .alias("state_done_flags")
    ])
    result
    return (result,)


@app.cell
def _(result):
    results =  result.select(["session_id", "items", "item_count", "action_lists", "watch_times", "state_done_flags"])
    results
    return (results,)


@app.cell
def _(results):
    results["action_lists"]
    return


@app.cell
def _(nn, torch):
    class GRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers=1):
            super(GRU, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )

        def forward(self, x, previous_hidden=None):
            if previous_hidden is None:
                batch_size = x.size(0)
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            else:
                hidden = previous_hidden
            output, hidden = self.gru(x, hidden)
            return output, hidden
    return (GRU,)


@app.cell
def _(GRU):
    state_encoder = GRU(input_dim=1171, hidden_dim=128, num_layers=1)
    state_encoder
    return (state_encoder,)


@app.cell
def _(TypedDict, torch):
    class DatasetItem(TypedDict):
        embeddings: torch.Tensor
        watch_times: torch.Tensor
        state_done_flags: torch.Tensor
        action_list: torch.Tensor
    return (DatasetItem,)


@app.cell
def _(item_embeddings, torch):
    tensor_embeddings = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in item_embeddings.items()
    }
    return (tensor_embeddings,)


@app.cell
def _(results, torch):
    action_lists = results["action_lists"].to_list()
    action_list = torch.tensor(action_lists[0], dtype=torch.long)
    return action_list, action_lists


@app.cell
def _(action_list):
    action_list.shape
    return


@app.cell
def _(Dataset, DatasetItem, Dict, pl, torch):
    class RecommendationDataset(Dataset):
        def __init__(self, df: pl.DataFrame, tensor_embeddings: Dict[int, torch.Tensor]) -> None:
            self.items = df["items"].to_list()
            self.watch_times = [torch.tensor(wt, dtype=torch.float32) for wt in df["watch_times"].to_list()]
            self.state_done_flags = [torch.tensor(sdf, dtype=torch.bool) for sdf in df["state_done_flags"].to_list()]
            self.action_lists = df["action_lists"].to_list()
            self.tensor_embeddings = tensor_embeddings

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> DatasetItem:
            embeddings = torch.stack([self.tensor_embeddings[item_id] for item_id in self.items[idx]])
            action_list = torch.tensor(self.action_lists[idx], dtype=torch.long)
            return {
                "embeddings": embeddings,  # [seq_len, embedding_dim]
                "watch_times": self.watch_times[idx],  # [seq_len]
                "state_done_flags": self.state_done_flags[idx],  # [seq_len]
                "action_list": action_list,  # [seq_len, action_length]
            }
    return (RecommendationDataset,)


@app.cell
def _(results):
    results["action_lists"]
    return


@app.cell
def _(RecommendationDataset, results, tensor_embeddings):
    dataset = RecommendationDataset(results, tensor_embeddings)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    print(dataset[0])
    return


@app.cell
def create_rl_batch(DatasetItem, Dict, List, torch):
    def create_rl_batch(batch_items: List[DatasetItem]) -> Dict[str, torch.Tensor]:
        all_states = []
        all_rewards = []
        all_dones = []
        all_actions = []

        for item in batch_items:
            embeddings = item["embeddings"]  # [seq_len, embedding_dim]
            watch_times = item["watch_times"]  # [seq_len]
            action_list = item["action_list"]  # [seq_len, action_length]

            done_flags = item["state_done_flags"]  # [seq_len]
            done_indices = torch.where(done_flags)[0]

            if len(done_indices) > 0:
                for i, done_idx in enumerate(done_indices):
                    done_idx = done_idx.item()

                    done_len = done_idx + 1

                    dones = torch.zeros(done_len, dtype=torch.bool)
                    dones[-1] = True

                    all_states.append(embeddings[:done_len]) # [seq_len[:done_len], embedding_dim]
                    all_rewards.append(watch_times[:done_len]) # [seq_len[:done_len]]
                    all_dones.append(dones)
                    all_actions.append(action_list[i])

            if len(done_indices) == 0:
                continue

        new_batch_size = len(all_states)

        max_len = max(states.shape[0] for states in all_states)
        embedding_dim = all_states[0].shape[1]

        padded_states = torch.zeros(new_batch_size, max_len, embedding_dim)
        padded_rewards = torch.zeros(new_batch_size, max_len)
        padded_dones = torch.zeros(new_batch_size, max_len, dtype=torch.bool)
        masks = torch.zeros(new_batch_size, max_len, dtype=torch.bool)
        padded_actions = torch.stack(all_actions)
        seq_lengths = torch.tensor([states.shape[0] for states in all_states], dtype=torch.long)

        for i in range(new_batch_size):
            seq_len = all_states[i].shape[0]
            padded_states[i, :seq_len] = all_states[i]
            padded_rewards[i, :seq_len] = all_rewards[i]
            padded_dones[i, :seq_len] = all_dones[i]
            masks[i, :seq_len] = True

        return {
            "state": padded_states,       # [new_batch_size, max_len, embedding_dim]
            "action": padded_actions,     # [new_batch_size, action_len]
            "reward": padded_rewards,     # [new_batch_size, max_len]
            "done": padded_dones,         # [new_batch_size, max_len]
            "mask": masks,                # [new_batch_size, max_len]
            "seq_lengths": seq_lengths    # [new_batch_size]
        }
    return (create_rl_batch,)


@app.cell
def _():
    # class GRUPolicy(nn.Module):
    #     def __init__(self, input_dim, hidden_dim, output_dim):
    #         super().__init__()
    #         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    #         self.fc = nn.Linear(hidden_dim, output_dim)

    #     def forward(self, states, masks, hidden=None):
    #         seq_lengths = masks.sum(dim=1).long()

    #         packed_states = nn.utils.rnn.pack_padded_sequence(
    #             states, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
    #         )

    #         packed_output, hidden = self.gru(packed_states, hidden)

    #         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

    #         action_logits = self.fc(output)

    #         return action_logits, hidden
    return


@app.cell
def _():
    # def compute_ppo_loss(states, actions, old_log_probs, returns, advantages, policy, value_net, masks):
    #     # 現在の方策と価値の評価
    #     action_logits, _ = policy(states, masks)
    #     values = value_net(states, masks)

    #     # マスクを適用して有効なタイムステップのみ計算
    #     action_logits = action_logits[masks]
    #     actions = actions[masks]
    #     old_log_probs = old_log_probs[masks]
    #     returns = returns[masks]
    #     advantages = advantages[masks]
    #     values = values[masks]

    #     # PPOの方策損失
    #     new_log_probs = F.log_softmax(action_logits, dim=-1)
    #     new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    #     ratio = torch.exp(new_log_probs - old_log_probs)
    #     surr1 = ratio * advantages
    #     surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    #     policy_loss = -torch.min(surr1, surr2).mean()

    #     # 価値損失
    #     value_loss = F.mse_loss(values, returns)

    #     return policy_loss + value_coef * value_loss
    return


@app.cell
def _():
    # class Environment:
    #     def reset(self):
    #         # 環境を初期状態にリセット
    #         return initial_state

    #     def step(self, action):
    #         # アクションを実行し、次の状態、報酬、完了フラグを返す
    #         return next_state, reward, done, info
    return


@app.cell
def _():
    # class Agent:
    #     def __init__(self, policy_network, value_network=None):
    #         self.policy_network = policy_network
    #         self.value_network = value_network

    #     def act(self, state):
    #         # 方策に基づいて行動を選択
    #         action_probs = self.policy_network(state)
    #         return sample_action(action_probs)

    #     def update(self, experiences):
    #         # 経験から学習
    #         loss = compute_loss(experiences)
    #         update_networks(loss)
    return


@app.cell
def ReplayBuffer():
    # class ReplayBuffer:
    #     def __init__(self, capacity):
    #         self.buffer = []
    #         self.capacity = capacity

    #     def add(self, episode):
    #         # エピソードを保存
    #         if len(self.buffer) >= self.capacity:
    #             self.buffer.pop(0)
    #         self.buffer.append(episode)

    #     def sample(self, batch_size):
    #         # バッチサイズ分のエピソードをサンプリング
    #         batch = random.sample(self.buffer, batch_size)
    #         return self.process_batch(batch)

    #     def process_batch(self, episodes):
    #         # エピソードをバッチ処理用に前処理
    #         return pad_and_mask_sequences(episodes)
    return


@app.cell
def _():
    # def train(agent, env, replay_buffer, num_episodes=1000):
    #     for episode in range(num_episodes):
    #         # エピソード収集
    #         state = env.reset()
    #         episode_data = []

    #         while True:
    #             action = agent.act(state)
    #             next_state, reward, done, _ = env.step(action)

    #             # 経験を記録
    #             episode_data.append(Experience(state, action, reward, next_state, done))

    #             state = next_state
    #             if done:
    #                 break

    #         # リプレイバッファに追加
    #         replay_buffer.add(episode_data)

    #         # バッチサンプリングと学習
    #         if len(replay_buffer) >= batch_size:
    #             batch = replay_buffer.sample(batch_size)
    #             agent.update(batch)
    return


@app.cell
def create_rl_batch():
    # def create_rl_batch(batch_items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    #     seq_lengths = [item["embeddings"].shape[0] for item in batch_items]
    #     max_seq_len = max(seq_lengths)
    #     batch_size = len(batch_items)
    #     embedding_dim = batch_items[0]["embeddings"].shape[1]

    #     batch_embeddings = torch.zeros(batch_size, max_seq_len, embedding_dim)
    #     batch_watch_times = torch.zeros(batch_size, max_seq_len)
    #     batch_done_flags = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    #     batch_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)

    #     for i, item in enumerate(batch_items):
    #         seq_len = seq_lengths[i]
    #         batch_embeddings[i, :seq_len] = item["embeddings"]
    #         batch_watch_times[i, :seq_len] = item["watch_times"]
    #         batch_done_flags[i, :seq_len] = item["state_done_flags"]
    #         batch_masks[i, :seq_len] = True

    #     states = []
    #     next_states = []
    #     rewards = []
    #     actions = []

    #     for i in range(batch_size):
    #         done_indices = torch.nonzero(batch_done_flags[i]).squeeze(-1)

    #         if done_indices.numel() > 0:
    #             for j, done_idx in enumerate(done_indices):
    #                 if done_idx == 0:
    #                     continue 
    #                 state_seq = batch_embeddings[i, :done_idx]
    #                 state_watch_times = batch_watch_times[i, :done_idx]
    #                 state = torch.cat([
    #                     state_seq,
    #                     state_watch_times.unsqueeze(-1)
    #                 ], dim=-1)
    #                 states.append(state)

    #                 print(len(states))

    #                 action = batch_embeddings[i, done_idx]
    #                 actions.append(action)

    #                 reward = batch_watch_times[i, done_idx]
    #                 rewards.append(reward)

    #                 if done_idx < seq_lengths[i] - 1:
    #                     next_state_seq = batch_embeddings[i, :(done_idx+1)]
    #                     next_state_watch_times = batch_watch_times[i, :(done_idx+1)]
    #                     next_state = torch.cat([
    #                         next_state_seq,
    #                         next_state_watch_times.unsqueeze(-1)
    #                     ], dim=-1)
    #                     next_states.append(next_state)
    #                 else:
    #                     next_states.append(None)

    #     max_state_len = max(s.shape[0] for s in states)
    #     padded_states = []
    #     padded_next_states = []
    #     state_masks = []

    #     for state, next_state in zip(states, next_states):
    #         state_len = state.shape[0]
    #         padded_state = torch.zeros(max_state_len, state.shape[1])
    #         padded_state[:state_len] = state
    #         padded_states.append(padded_state)

    #         mask = torch.zeros(max_state_len, dtype=torch.bool)
    #         mask[:state_len] = True
    #         state_masks.append(mask)

    #         if next_state is not None:
    #             next_state_len = next_state.shape[0]
    #             padded_next_state = torch.zeros(max_state_len, next_state.shape[1])
    #             padded_next_state[:next_state_len] = next_state
    #             padded_next_states.append(padded_next_state)
    #         else:
    #             padded_next_states.append(torch.zeros(max_state_len, state.shape[1]))

    #     return {
    #         "state": torch.stack(padded_states),  # [N, max_state_len, embedding_dim+1]
    #         "action": torch.stack(actions),  # [N, embedding_dim]
    #         "reward": torch.stack(rewards),  # [N]
    #         "next_state": torch.stack(padded_next_states),  # [N, max_state_len, embedding_dim+1]
    #         "mask": torch.stack(state_masks),  # [N, max_state_len]
    #         "terminal": torch.tensor([ns is None for ns in next_states], dtype=torch.bool)  # [N]
    #     }
    return


@app.cell
def _(create_rl_batch, partial):
    collate_fn = partial(create_rl_batch)
    return (collate_fn,)


@app.cell
def _(DataLoader, collate_fn, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return (dataloader,)


@app.cell
def _(dataloader):
    dataloader
    return


@app.cell
def _(dataloader):
    bat = next(iter(dataloader))
    bat
    return (bat,)


@app.cell
def _(nn):
    class GRUNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super(GRUNetwork, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, seq_lengths, hidden=None):
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed_x, hidden) # [num_layers, new_batch_size, hidden_dim]
            current_state = hidden[-1]  # [new_batch_size, hidden_dim]
            output = self.fc(current_state)  # [new_batch_size, output_dim]
        
            return output, hidden
    return (GRUNetwork,)


@app.cell
def _():
    from collections import deque
    return (deque,)


@app.cell
def _():
    # class ReplayBuffer:
    #     def __init__(self, capacity=10000):
    #         self.buffer = deque(maxlen=capacity)

    #     def push(self, state, action, reward, next_state, done, mask):
    #         self.buffer.append((state, action, reward, next_state, done, mask))

    #     def sample(self, batch_size):
    #         batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
    #         state, action, reward, next_state, done, mask = zip(*batch)

    #         # リストからテンソルに変換
    #         return (
    #             torch.cat(state, dim=0),
    #             torch.cat(action, dim=0),
    #             torch.cat(reward, dim=0),
    #             torch.cat(next_state, dim=0),
    #             torch.cat(done, dim=0),
    #             torch.cat(mask, dim=0)
    #         )

    #     def __len__(self):
    #         return len(self.buffer)
    return


@app.cell
def _(nn):
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(PolicyNetwork, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, state):
            return self.fc(state)
    return (PolicyNetwork,)


@app.cell
def _():
    embedding_dim = 1171
    hidden_dim =128
    output_dim = 1171
    action_dim = 10
    return action_dim, embedding_dim, hidden_dim, output_dim


@app.cell
def _(
    GRU,
    PolicyNetwork,
    ReplayBuffer,
    action_dim,
    embedding_dim,
    gru_model,
    hidden_dim,
    output_dim,
):
    gru = GRU(embedding_dim, hidden_dim, output_dim)
    policy_network = PolicyNetwork(hidden_dim, action_dim)

    # オプティマイザの設定
    # optimizer = optim.Adam(list(gru_model.parameters()) + list(policy_network.parameters()), lr=lr)

    # ReplayBufferの初期化
    replay_buffer = ReplayBuffer()

    # モデルをトレーニングモードに設定
    gru_model.train()
    policy_network.train()
    return gru, policy_network, replay_buffer


@app.cell
def _():
    temperature = 1.0
    num_actions = 1000000
    learning_rate = 1e-3
    return learning_rate, num_actions, temperature


@app.cell
def _(Categorical, F, nn, torch):
    class ActorNetwork(nn.Module):
        def __init__(self, input_dim, action_dim, hidden_size, init_w=0):
            super().__init__() 

            self.linear1 = nn.Linear(input_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, action_dim)

            self.saved_log_probs = []
            self.rewards = []
            self.correction = []
            self.lambda_k = []

            self.action_source = {"pi": "pi", "beta": "beta"}
            self.select_action = self._select_action_with_TopK_correction

        def forward(self, inputs):
            x = inputs
            x = F.relu(self.linear1(x))
            action_scores = self.linear2(x)
            return F.softmax(action_scores, dim=1)

        def pi_beta_sample(self, state, beta, action):

            beta_probs = beta(state.detach(), action=action)
            pi_probs = self.forward(state)

            beta_categorical = Categorical(beta_probs)
            pi_categorical = Categorical(pi_probs)

            available_actions = {
                "pi": pi_categorical.sample(),
                "beta": beta_categorical.sample(),
            }
            pi_action = available_actions[self.action_source["pi"]]
            beta_action = available_actions[self.action_source["beta"]]

            pi_log_prob = pi_categorical.log_prob(pi_action)
            beta_log_prob = beta_categorical.log_prob(beta_action)

            return pi_log_prob, beta_log_prob, pi_probs


        def _select_action_with_TopK_correction(self, state, beta, action, K):
            pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

            corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)

            l_k = K * (1 - torch.exp(pi_log_prob)) ** (K - 1)

            self.correction.append(corr)
            self.lambda_k.append(l_k)
            self.saved_log_probs.append(pi_log_prob)

            return pi_probs

        def gc(self):
            del self.rewards[:]
            del self.saved_log_probs[:]
            del self.correction[:]
            del self.lambda_k[:]
    return (ActorNetwork,)


@app.cell
def _(action_dim, embedding_dim, nn):
    item_embeddings  = nn.Embedding(action_dim, embedding_dim)
    return (item_embeddings,)


@app.cell
def _(embedding_dim, hidden_dim, nn):
    behavior_projection = nn.Linear(hidden_dim, embedding_dim)
    return (behavior_projection,)


@app.cell
def _(torch):
    def compute_top_k_policy(policy_prob, K):
        # α_θ(a|s) = 1 - (1 - π_θ(a|s))^K
        top_k_policy = 1 - (1 - policy_prob) ** K
        return top_k_policy

    def compute_top_k_multiplier(policy_prob, actions, K):
        # λ_K(s_t, a_t) = K(1-π_θ(a_t|s_t))^(K-1)
        action_probs = torch.gather(policy_prob, 1, actions)
        multiplier = K * (1 - action_probs) ** (K-1)
        return multiplier
    return compute_top_k_multiplier, compute_top_k_policy


@app.cell
def _(
    F,
    K,
    behavior_projection,
    cap,
    dataloader,
    gru,
    item_embeddings,
    main_policy,
    model,
    optimizer,
    temperature,
    torch,
):
    for ba in dataloader:
        states = ba["state"]          # [new_batch_size, max_len, embedding_dim]
        actions = ba["action"]        # [new_batch_size, action_len]
        rewards = ba["reward"]        # [new_batch_size, max_len]
        dones = ba["done"]            # [new_batch_size, max_len]
        masks = ba["mask"]            # [new_batch_size, max_len]
        seq_lengths = ba["seq_lengths"]  # [new_batch_size]

        batch_size = states.size(0)
        max_len = states.size(1)

        states, _ = gru(states, seq_lengths)  # [new_batch_size, output_dim]
        logits = torch.matmul(states, item_embeddings.weight.t()) / temperature
        policy_probs = F.softmax(logits, dim=1) # [batch_size, num_actions]
    
        b_states = behavior_projection(states.detach())
        b_logits = torch.matmul(b_states, item_embeddings.weight.t())
        b_policy_probs = F.softmax(b_logits, dim=1)

        action_probs = torch.gather(policy_probs, 1, actions)
        b_action_probs = torch.gather(b_policy_probs, 1, actions)
        importance_weights = action_probs / (b_action_probs + 1e-8)

        importance_weights = torch.clamp(importance_weights, max=cap)
    
        top_k_multiplier = model.compute_top_k_multiplier(main_policy, actions, K)

        log_probs = torch.log(action_probs + 1e-8)
        loss = -torch.mean(importance_weights * top_k_multiplier * rewards * log_probs)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()
    return (
        action_probs,
        actions,
        b_action_probs,
        b_logits,
        b_policy_probs,
        b_states,
        ba,
        batch_size,
        dones,
        importance_weights,
        log_probs,
        logits,
        loss,
        masks,
        max_len,
        policy_probs,
        rewards,
        seq_lengths,
        states,
        top_k_multiplier,
        total_loss,
    )


@app.cell
def _(replay_buffer):
    sample = replay_buffer.sample(batch_size=2)
    sample
    return (sample,)


@app.cell
def _():
    # train_dataset = RecommendationDataset(train_df, tensor_embeddings)
    # test_dataset = RecommendationDataset(test_df, tensor_embeddings)
    return


@app.cell
def _():
    # for i, (rec_items, rec_time) in enumerate(zip(recommendations, rec_times)):
    #     # Get viewing history up to this recommendation time
    #     history_up_to_rec = session_views.filter(pl.col("created_at") < rec_time)

    #     if len(history_up_to_rec) == 0:
    #         # Skip if no viewing history before this recommendation
    #         continue

    #     # Extract movie IDs and watch times
    #     history_items = history_up_to_rec["movie_id"].to_list()
    #     watch_times = history_up_to_rec["watch_time"].to_list()

    #     history_embeddings = []
    #     for item_id in history_items:
    #         if str(item_id) in self.item_embeddings:
    #             embedding = self.item_embeddings[str(item_id)]
    #             history_embeddings.append(embedding)

    #     history_tensor = torch.stack(history_embeddings).unsqueeze(0).to(self.device)
    return


@app.cell
def _():
    # _, hidden = self.state_encoder(history_tensor, previous_hidden)
    return


@app.cell
def _():
    # class RLRecommendationDataset(Dataset):
    #     """Dataset for reinforcement learning with sequential state updates"""
    #     def __init__(self, session_data, recommendation_data, item_embeddings, 
    #                  state_encoder, hidden_dim, device="cpu"):
    #         self.session_data = session_data
    #         self.recommendation_data = recommendation_data
    #         self.item_embeddings = item_embeddings
    #         self.state_encoder = state_encoder.to(device)
    #         self.hidden_dim = hidden_dim
    #         self.device = device

    #         # Process data to create sequence of (state, action, reward, next_state) tuples
    #         self.rl_data = self._prepare_rl_data()

    #     def _prepare_rl_data(self):
    #         rl_data = []

    #         # Group recommendation data by session_id
    #         grouped_recommendations = self.recommendation_data.group_by("session_id").agg([
    #             pl.col("movie_ids").alias("recommendations"),
    #             pl.col("created_at").alias("recommendation_times")
    #         ])

    #         # Process each session
    #         for session_row in grouped_recommendations.iter_rows(named=True):
    #             session_id = session_row["session_id"]
    #             recommendations = session_row["recommendations"]
    #             rec_times = session_row["recommendation_times"]

    #             # Get viewing history for this session
    #             session_views = self.session_data.filter(pl.col("session_id") == session_id)

    #             if len(session_views) == 0 or len(recommendations) == 0:
    #                 continue

    #             previous_hidden = None
    #             previous_state = None

    #             # Process each recommendation in temporal order
    #             for i, (rec_items, rec_time) in enumerate(zip(recommendations, rec_times)):
    #                 # Get viewing history up to this recommendation time
    #                 history_up_to_rec = session_views.filter(pl.col("created_at") < rec_time)

    #                 if len(history_up_to_rec) == 0:
    #                     # Skip if no viewing history before this recommendation
    #                     continue

    #                 # Extract movie IDs and watch times
    #                 history_items = history_up_to_rec["movie_id"].to_list()
    #                 watch_times = history_up_to_rec["watch_time"].to_list()

    #                 # Get embeddings for history items
    #                 history_embeddings = []
    #                 for item_id in history_items:
    #                     if str(item_id) in self.item_embeddings:
    #                         embedding = self.item_embeddings[str(item_id)]
    #                         history_embeddings.append(embedding)

    #                 if not history_embeddings:
    #                     continue

    #                 # Create tensor from embeddings
    #                 history_tensor = torch.stack(history_embeddings).unsqueeze(0).to(self.device)

    #                 # Use GRU to get state representation
    #                 with torch.no_grad():
    #                     if previous_hidden is not None:
    #                         # Update state using previous hidden state
    #                         _, hidden = self.state_encoder(history_tensor, previous_hidden)
    #                         state = hidden[-1]
    #                     else:
    #                         # Create new state
    #                         _, hidden = self.state_encoder(history_tensor)
    #                         state = hidden[-1]

    #                     previous_hidden = hidden

    #                 # Get recommended items as actions
    #                 recommended_items = [int(item) for item in rec_items.split(",")]

    #                 # Find next viewing history (for reward calculation)
    #                 if i < len(rec_times) - 1:
    #                     next_rec_time = rec_times[i + 1]
    #                     next_history = session_views.filter(
    #                         (pl.col("created_at") >= rec_time) & 
    #                         (pl.col("created_at") < next_rec_time)
    #                     )
    #                 else:
    #                     next_history = session_views.filter(pl.col("created_at") >= rec_time)

    #                 # Calculate reward (e.g., based on watch times of recommended items)
    #                 watched_recommended = next_history.filter(
    #                     pl.col("movie_id").is_in(recommended_items)
    #                 )
    #                 reward = len(watched_recommended) / len(recommended_items) if recommended_items else 0

    #                 # Store as RL tuple
    #                 rl_data.append({
    #                     "session_id": session_id,
    #                     "state": state.cpu().numpy(),
    #                     "action": recommended_items,
    #                     "reward": reward,
    #                     "previous_state": previous_state.cpu().numpy() if previous_state is not None else None,
    #                     "timestamp": rec_time
    #                 })

    #                 previous_state = state

    #         return rl_data

    #     def __len__(self):
    #         return len(self.rl_data)

    #     def __getitem__(self, idx):
    #         return self.rl_data[idx]
    return


@app.cell
def create_rl_batch():
    # def create_rl_batch(batch, embedding_dim, max_sequence_length):
    #     """Collate function for creating batches of RL data"""
    #     states = torch.tensor(np.stack([item["state"] for item in batch]), dtype=torch.float32)
    #     actions = [item["action"] for item in batch]
    #     rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float32)

    #     # Handle previous states (could be None for initial states)
    #     previous_states = []
    #     for item in batch:
    #         if item["previous_state"] is not None:
    #             previous_states.append(item["previous_state"])
    #         else:
    #             previous_states.append(np.zeros(states.shape[1]))
    #     previous_states = torch.tensor(np.stack(previous_states), dtype=torch.float32)

    #     return {
    #         "states": states,
    #         "actions": actions,
    #         "rewards": rewards,
    #         "previous_states": previous_states
    #     }
    return


@app.cell
def _():
    # def create_rl_batch(
    #     batch: List[Dict[str, torch.Tensor]], embedding_dim, max_sequence_length: int = 50
    # ) -> Dict[str, torch.Tensor]:
    #     if not batch:
    #         raise ValueError("Batch cannot be empty")

    #     batch_size = len(batch)  # N

    #     padded_embeddings = torch.zeros(
    #         batch_size, max_sequence_length, embedding_dim
    #     )  # [N, max_sequence_length, embedding_dim]

    #     padded_watch_times = torch.zeros(
    #         batch_size, max_sequence_length
    #     )  # [N, max_sequence_length]

    #     mask = torch.zeros(
    #         batch_size, max_sequence_length, dtype=torch.bool
    #     )  # [N, max_sequence_length]

    #     for i, sample in enumerate(batch):
    #         sequence_length = min(sample["embeddings"].size(0), max_sequence_length)
    #         padded_embeddings[i, :sequence_length] = sample["embeddings"][
    #             :sequence_length
    #         ]  # [sequence_length, embedding_dim]
    #         padded_watch_times[i, :sequence_length] = sample["watch_times"][
    #             :sequence_length
    #         ]  # [sequence_length]
    #         mask[i, :sequence_length] = 1  # [sequence_length]

    #     state = torch.cat(
    #         [padded_embeddings, padded_watch_times.unsqueeze(-1)], dim=-1
    #     )  # [N, max_sequence_length, embedding_dim + 1]

    #     next_state = torch.zeros_like(state)  # [N, max_sequence_length, embedding_dim + 1]
    #     next_state[:, :-1] = state[:, 1:]

    #     true_embedding = torch.stack(
    #         [b["embeddings"][-1] for b in batch]
    #     )  # [N, embedding_dim]

    #     true_watch_times = torch.stack([b["watch_times"][-1] for b in batch])  # [N]

    #     true_watch_times = (true_watch_times - true_watch_times.mean()) / (
    #         true_watch_times.std() + 1e-8
    #     )  # [N]

    #     return {
    #         "state": state,  # [N, max_sequence_length, embedding_dim + 1]
    #         "action": true_embedding,  # [N, embedding_dim]
    #         "reward": true_watch_times,  # [N]
    #         "next_state": next_state,  # [N, max_sequence_length, embedding_dim + 1]
    #         "mask": mask,  # [N, max_sequence_length]
    #     }
    return


if __name__ == "__main__":
    app.run()
