from typing import Dict, TypedDict

import polars as pl
import torch
from torch.utils.data import Dataset


class DatasetItem(TypedDict):
    embeddings: torch.Tensor
    watch_times: torch.Tensor


class RecommendationDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, tensor_embeddings: Dict[int, torch.Tensor]
    ) -> None:
        self.items = df["items"].to_list()
        self.watch_times = [
            torch.tensor(wt, dtype=torch.float32) for wt in df["watch_times"].to_list()
        ]
        self.tensor_embeddings = tensor_embeddings

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DatasetItem:
        embeddings = torch.stack(
            [self.tensor_embeddings[item_id] for item_id in self.items[idx]]
        )
        return {
            "embeddings": embeddings,
            "watch_times": self.watch_times[idx],
        }
