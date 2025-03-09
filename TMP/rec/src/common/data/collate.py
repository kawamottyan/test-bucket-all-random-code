from typing import Dict, List

import torch


def create_rl_batch(
    batch: List[Dict[str, torch.Tensor]], embedding_dim, max_sequence_length: int = 50
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Batch cannot be empty")

    required_keys = {"embeddings", "watch_times"}
    for sample in batch:
        missing_keys = required_keys - set(sample.keys())
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")

    batch_size = len(batch)  # N

    padded_embeddings = torch.zeros(
        batch_size, max_sequence_length, embedding_dim
    )  # [N, max_sequence_length, embedding_dim]

    padded_watch_times = torch.zeros(
        batch_size, max_sequence_length
    )  # [N, max_sequence_length]

    mask = torch.zeros(
        batch_size, max_sequence_length, dtype=torch.bool
    )  # [N, max_sequence_length]

    for i, sample in enumerate(batch):
        sequence_length = min(sample["embeddings"].size(0), max_sequence_length)
        padded_embeddings[i, :sequence_length] = sample["embeddings"][
            :sequence_length
        ]  # [sequence_length, embedding_dim]
        padded_watch_times[i, :sequence_length] = sample["watch_times"][
            :sequence_length
        ]  # [sequence_length]
        mask[i, :sequence_length] = 1  # [sequence_length]

    state = torch.cat(
        [padded_embeddings, padded_watch_times.unsqueeze(-1)], dim=-1
    )  # [N, max_sequence_length, embedding_dim + 1]

    next_state = torch.zeros_like(state)  # [N, max_sequence_length, embedding_dim + 1]
    next_state[:, :-1] = state[:, 1:]

    true_embedding = torch.stack(
        [b["embeddings"][-1] for b in batch]
    )  # [N, embedding_dim]

    true_watch_times = torch.stack([b["watch_times"][-1] for b in batch])  # [N]

    true_watch_times = (true_watch_times - true_watch_times.mean()) / (
        true_watch_times.std() + 1e-8
    )  # [N]

    return {
        "state": state,  # [N, max_sequence_length, embedding_dim + 1]
        "action": true_embedding,  # [N, embedding_dim]
        "reward": true_watch_times,  # [N]
        "next_state": next_state,  # [N, max_sequence_length, embedding_dim + 1]
        "mask": mask,  # [N, max_sequence_length]
    }
