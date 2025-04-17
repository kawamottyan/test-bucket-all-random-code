import logging
import os
from functools import partial
from typing import Any, Dict, List, Set, TypedDict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset

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


def generate_state_flags(row):
    item_count = row["item_count"]
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


class DatasetItem(TypedDict):
    embeddings: torch.Tensor
    watch_times: torch.Tensor
    state_done_flags: torch.Tensor
    action_list: torch.Tensor


class RecommendationDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, tensor_embeddings: Dict[int, torch.Tensor]
    ) -> None:
        self.items = df["items"].to_list()
        self.watch_times = [
            torch.tensor(wt, dtype=torch.float32) for wt in df["watch_times"].to_list()
        ]
        self.state_done_flags = [
            torch.tensor(sdf, dtype=torch.bool)
            for sdf in df["state_done_flags"].to_list()
        ]
        self.action_lists = df["action_lists"].to_list()
        self.tensor_embeddings = tensor_embeddings

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        embeddings = torch.stack(
            [self.tensor_embeddings[item_id] for item_id in self.items[idx]]
        )
        action_list = torch.tensor(self.action_lists[idx], dtype=torch.long)
        return {
            "embeddings": embeddings,  # [seq_len, embedding_dim]
            "watch_times": self.watch_times[idx],  # [seq_len]
            "state_done_flags": self.state_done_flags[idx],  # [seq_len]
            "action_list": action_list,  # [seq_len, action_length]
        }


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

                all_states.append(
                    embeddings[:done_len]
                )  # [seq_len[:done_len], embedding_dim]
                all_rewards.append(watch_times[:done_len])  # [seq_len[:done_len]]
                all_dones.append(dones)
                all_actions.append(action_list[i])

        if len(done_indices) == 0:
            continue

    new_batch_size = len(all_states)

    max_seq_len = max(states.shape[0] for states in all_states)
    embedding_dim = all_states[0].shape[1]

    padded_states = torch.zeros(new_batch_size, max_seq_len, embedding_dim)
    padded_rewards = torch.zeros(new_batch_size, max_seq_len)
    padded_dones = torch.zeros(new_batch_size, max_seq_len, dtype=torch.bool)
    masks = torch.zeros(new_batch_size, max_seq_len, dtype=torch.bool)
    padded_actions = torch.stack(all_actions)
    seq_lengths = torch.tensor(
        [states.shape[0] for states in all_states], dtype=torch.long
    )

    for i in range(new_batch_size):
        seq_len = all_states[i].shape[0]
        padded_states[i, :seq_len] = all_states[i]
        padded_rewards[i, :seq_len] = all_rewards[i]
        padded_dones[i, :seq_len] = all_dones[i]
        masks[i, :seq_len] = True

    return {
        "state": padded_states,  # [new_batch_size, max_seq_len, embedding_dim]
        "action": padded_actions,  # [new_batch_size, action_len]
        "reward": padded_rewards,  # [new_batch_size, max_seq_len]
        "done": padded_dones,  # [new_batch_size, max_seq_len]
        "mask": masks,  # [new_batch_size, max_seq_len]
        "seq_lengths": seq_lengths,  # [new_batch_size]
    }


class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, seq_lengths):
        batch_size = x.size(0)
        hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=x.device
        )

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(
            packed_x, hidden
        )  # [num_layers, new_batch_size, hidden_dim]
        current_state = hidden[-1]  # [new_batch_size, hidden_dim]
        output = self.fc(current_state)  # [new_batch_size, output_dim]

        return output, hidden


class ReinforceRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, temperature=1, K=10):
        super().__init__()
        # 共有RNNネットワーク（GRUを使用）
        self.rnn = GRUNetwork(input_dim, hidden_dim, hidden_dim)

        # 入力用アイテム埋め込み (論文のU)
        self.item_embeddings_input = nn.Embedding(action_dim, input_dim)

        # 出力用アイテム埋め込み (論文のV)
        self.item_embeddings_output = nn.Embedding(action_dim, hidden_dim)

        # メインポリシー用レイヤー
        self.main_policy_layer = nn.Linear(hidden_dim, hidden_dim)

        # 行動ポリシー用レイヤー
        self.behavior_policy_layer = nn.Linear(hidden_dim, hidden_dim)

        self.temperature = temperature
        self.K = K
        self.action_dim = action_dim

    def forward(self, states, seq_lengths, actions=None):
        # RNNを通じてユーザー状態を取得
        user_states, _ = self.rnn(states, seq_lengths)  # [batch_size, hidden_dim]

        # メインポリシー計算
        main_features = self.main_policy_layer(user_states)
        main_logits = (
            torch.matmul(main_features, self.item_embeddings_output.weight.t())
            / self.temperature
        )
        main_probs = F.softmax(main_logits, dim=1)  # [batch_size, action_dim]

        # 行動ポリシー計算 - 勾配をブロック（論文のFigure 1に対応）
        behavior_states = user_states.detach()  # 勾配をブロック
        behavior_features = self.behavior_policy_layer(behavior_states)
        behavior_logits = torch.matmul(
            behavior_features, self.item_embeddings_output.weight.t()
        )
        behavior_probs = F.softmax(behavior_logits, dim=1)  # [batch_size, action_dim]

        main_action_probs = torch.gather(main_probs, 1, actions)
        behavior_action_probs = torch.gather(behavior_probs, 1, actions)
        return main_probs, behavior_probs, main_action_probs, behavior_action_probs

    def compute_top_k_probs(self, probs):
        """
        アイテムがTop-Kに出現する確率を計算
        論文の式: a_θ(a|s) = 1-(1-π_θ(a|s))^K
        """
        top_k_probs = 1 - (1 - probs) ** self.K
        return top_k_probs

    def compute_top_k_multiplier(self, probs, actions):
        """
        Top-K off-policyの乗数λ_K(s_t, a_t)を計算
        論文の式(8): λ_K(s_t, a_t) = K(1-π_θ(a_t|s_t))^(K-1)
        """
        action_probs = torch.gather(probs, 1, actions)
        multiplier = self.K * (1 - action_probs) ** (self.K - 1)
        return multiplier

    def boltzmann_exploration(self, probs, M=None, K_greedy=None):
        """
        Boltzmann探索の実装(論文のSection 5)
        M: 候補アイテム数(通常はKより大きい)
        K_greedy: 確定的に選択するトップアイテム数
        """
        if M is None:
            M = min(self.K * 10, self.action_dim)  # デフォルトはK*10

        if K_greedy is None:
            K_greedy = self.K // 2  # デフォルトはK/2

        batch_size = probs.size(0)
        device = probs.device

        # トップM項目を取得（効率的な近似最近傍検索の代わり）
        _, top_m_indices = torch.topk(probs, M, dim=1)  # [batch_size, M]

        # トップM項目の確率を取得
        batch_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, M)
        )
        top_m_probs = probs[batch_indices, top_m_indices]  # [batch_size, M]

        # 結果格納用
        recommendations = torch.zeros(
            batch_size, self.K, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            # トップK_greedy項目を確定的に選択
            _, greedy_indices = torch.topk(top_m_probs[i], K_greedy)
            recommendations[i, :K_greedy] = top_m_indices[i, greedy_indices]

            # 残りの項目からサンプリング
            mask = torch.ones(M, dtype=torch.bool, device=device)
            mask[greedy_indices] = False

            remaining_indices = top_m_indices[i, mask]
            remaining_probs = top_m_probs[i, mask]

            # 確率を正規化
            remaining_probs = remaining_probs / remaining_probs.sum()

            # 残りのK-K_greedy個をサンプリング
            if self.K > K_greedy and len(remaining_indices) > 0:
                sample_size = min(self.K - K_greedy, len(remaining_indices))
                sampled_idx = torch.multinomial(
                    remaining_probs, sample_size, replacement=False
                )
                recommendations[i, K_greedy : K_greedy + sample_size] = (
                    remaining_indices[sampled_idx]
                )

        return recommendations


def train_step(model, batch, optimizer, weight_cap=np.exp(3)):
    states = batch["state"]
    actions = batch["action"]
    rewards = batch["reward"].sum(dim=1, keepdim=True)  # 累積報酬
    seq_lengths = batch["seq_lengths"]

    # 両方のポリシーを計算（パラメータを共有しながら）
    main_probs, behavior_probs, main_action_probs, behavior_action_probs = model(
        states, seq_lengths, actions
    )

    # Top-K確率とOff-Policy補正
    top_k_probs = model.compute_top_k_probs(main_probs)  # α_θ(a|s) = 1-(1-π_θ(a|s))^K

    # 重要度重み付け
    importance_weights = main_action_probs / (behavior_action_probs + 1e-8)
    importance_weights = torch.clamp(importance_weights, max=weight_cap)

    # Top-K乗数
    top_k_multiplier = model.compute_top_k_multiplier(main_probs, actions)

    # 損失計算 - 式(7)に基づくTop-K Off-Policy勾配
    log_probs = torch.log(main_action_probs + 1e-8)
    loss = -torch.mean(importance_weights * top_k_multiplier * rewards * log_probs)

    # 最適化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def recommend_items(model, states, seq_lengths, use_exploration=True):
    """
    アイテムの推薦を行う関数
    use_exploration: True=Boltzmann探索、False=確定的なトップK
    """
    # ポリシーの計算
    main_probs, _ = model(states, seq_lengths)

    if use_exploration:
        # Boltzmann探索を使用（論文Section 5）
        recommendations = model.boltzmann_exploration(main_probs)
    else:
        # 確定的にトップKを選択
        _, recommendations = torch.topk(main_probs, model.K, dim=1)

    return recommendations


def main() -> None:
    try:
        s3_handler: S3Handler = get_s3_handler(config_dict)

        consented_uuids = get_consented_uuids()
        if not consented_uuids:
            logger.warning("No consented UUIDs found.")

        poster_viewed_df = s3_handler.get_all_logs(
            log_name="poster_viewed",
            columns=["uuid", "session_id", "movie_id", "watch_time", "created_at"],
        )

        assert poster_viewed_df is not None, "trainning data is None"

        recommendation_df = s3_handler.get_all_logs(
            log_name="recommendation",
            columns=["uuid", "session_id", "movie_ids", "created_at"],
        )

        assert recommendation_df is not None, "recommendation data is None"

        # todo
        result_df = (
            recommendation_df.sort("created_at")
            .with_columns([pl.arange(0, pl.count()).over("session_id").alias("row_nr")])
            .with_columns(pl.col("row_nr").max().over("session_id").alias("max_row_nr"))
            .filter(
                (pl.col("row_nr") == 1) | (pl.col("row_nr") == pl.col("max_row_nr"))
            )
            .drop(["row_nr", "max_row_nr"])
        )

        train_df = poster_viewed_df.filter(pl.col("uuid").is_in(consented_uuids))
        logger.info("Filtered data size after consent check: %d rows", train_df.height)

        unique_item_ids = (
            train_df.select(pl.col("movie_id").cast(pl.Utf8))
            .unique()
            .to_series()
            .to_list()
        )

        item_embeddings = {}
        embedding_batch_size = config_dict["data"]["embedding_batch_size"]
        total_batch_size = (
            len(unique_item_ids) + embedding_batch_size - 1
        ) // embedding_batch_size

        logger.info(
            "Processing %d items in %d batches...",
            len(unique_item_ids),
            total_batch_size,
        )
        total_batch_size = 10  # todo: remove this

        for i in range(total_batch_size):
            start_idx = i * embedding_batch_size
            end_idx = min((i + 1) * embedding_batch_size, len(unique_item_ids))
            batch = unique_item_ids[start_idx:end_idx]

            response = index.fetch(ids=list(batch), namespace="movies")

            vectors_dict = {
                int(id): vector.values for id, vector in response.vectors.items()
            }
            item_embeddings.update(vectors_dict)

            logger.info(
                "Batch %d/%d completed. Total embeddings: %d",
                i + 1,
                total_batch_size,
                len(item_embeddings),
            )

        valid_items = set(item_embeddings.keys())
        filtered_df = train_df.filter(pl.col("movie_id").is_in(valid_items))

        sorted_df = filtered_df.sort(["session_id", "created_at"])

        grouped_df = sorted_df.group_by("session_id").agg(
            [
                pl.col("movie_id").alias("items"),
                pl.col("watch_time").alias("watch_times"),
                pl.col("created_at").alias("item_created_ats"),
                pl.len().alias("item_count"),
            ]
        )
        min_item_size = config_dict["data"]["min_items"]
        grouped_df = grouped_df.filter(pl.col("items").list.len() >= 50 + 1)

        overall_max_id = result_df.select(pl.col("movie_ids").list.max().max()).item()

        grouped_recommendation_df = result_df.group_by("session_id").agg(
            [
                pl.col("movie_ids").alias("action_lists"),
                pl.col("created_at").alias("rec_created_ats"),
                pl.len().alias("rec_count"),
            ]
        )

        joined_df = grouped_df.join(
            grouped_recommendation_df, on="session_id", how="inner"
        )

        result = joined_df.with_columns(
            [
                pl.struct(
                    [
                        "item_count",
                        "item_created_ats",
                        "rec_created_ats",
                    ]
                )
                .map_elements(generate_state_flags, return_dtype=pl.List(pl.Boolean))
                .alias("state_done_flags")
            ]
        )

        train_df = result.select(
            [
                "session_id",
                "items",
                "item_count",
                "action_lists",
                "watch_times",
                "state_done_flags",
            ]
        )

        tensor_embeddings = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in item_embeddings.items()
        }

        train_dataset = RecommendationDataset(train_df, tensor_embeddings)

        collate_fn = partial(create_rl_batch)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        max_in_valid_items = max(valid_items)
        embedding_dim = 1171
        hidden_dim = 128
        output_dim = 1171
        action_dim = max(max_in_valid_items, overall_max_id) + 1
        temperature = 1.0
        learning_rate = 1e-3
        weight_cap = 5.0
        num_epochs = 5

        model = ReinforceRecommender(
            embedding_dim, hidden_dim, action_dim, temperature=1.0, K=10
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # トレーニング
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                loss = train_step(model, batch, optimizer)
                total_loss += loss
                print(loss)
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")

    except Exception as e:
        logger.exception("Error during ddpg model training: %s", str(e))


if __name__ == "__main__":
    main()
