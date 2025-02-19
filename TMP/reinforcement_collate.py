from typing import List, Dict, Any, Sequence
import numpy as np
import torch


def rolling_window(arr: Sequence[Any], window_size: int) -> np.ndarray:
    return np.array(
        [arr[i : i + window_size] for i in range(len(arr) - window_size + 1)]
    )


def apply_rolling_window(arrays: List[Sequence[Any]], window_size: int) -> np.ndarray:
    rolling_windows = [rolling_window(array, window_size) for array in arrays]
    return np.concatenate(rolling_windows, axis=0)


def apply_rolling_window_and_tensorize(
    batch_list: List[Sequence[Any]], frame_size: int, dtype: torch.dtype
) -> torch.Tensor:
    batch = apply_rolling_window(batch_list, frame_size + 1)
    return torch.tensor(batch, dtype=dtype)


def pad_collate(
    batch: List[Dict[str, torch.Tensor]], max_sequence_length: int = 50
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    embedding_dim = 1181

    # 固定長のテンソルを初期化
    padded_embeddings = torch.zeros(batch_size, max_sequence_length, embedding_dim)
    padded_watch_times = torch.zeros(batch_size, max_sequence_length)
    mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.bool)

    # パディングと必要に応じて切り捨て
    for i, sample in enumerate(batch):
        seq_len = min(sample["embeddings"].size(0), max_sequence_length)
        padded_embeddings[i, :seq_len] = sample["embeddings"][:seq_len]
        padded_watch_times[i, :seq_len] = sample["watch_times"][:seq_len]
        mask[i, :seq_len] = 1

    # 入力特徴量の結合
    state = torch.cat(
        [padded_embeddings, padded_watch_times.unsqueeze(-1)], dim=-1
    )  # [batch_size, max_sequence_length, 1182]

    # 次のstateを作成
    next_state = torch.zeros_like(state)
    next_state[:, :-1] = state[:, 1:]

    true_embedding = torch.stack([b["embeddings"][-1] for b in batch])
    true_watch_times = torch.stack([b["watch_times"][-1] for b in batch])
    true_watch_times = (true_watch_times - true_watch_times.mean()) / (
        true_watch_times.std() + 1e-8
    )

    return {
        "state": state,  # [batch_size, 50, 1182]  # 50は固定のmax_sequence_length
        "action": true_embedding,  # [batch_size, 1181]  # 次のアイテムのembedding
        "reward": true_watch_times,  # [batch_size]  # 各サンプルの視聴時間
        "next_state": next_state,  # [batch_size, 50, 1182]  # stateと同じ形状
        "mask": mask,  # [batch_size, 50]  # 有効なシーケンス部分を示すマスク
    }
