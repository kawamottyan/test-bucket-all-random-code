from typing import List

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ):
        super(ActorNetwork, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dims[0], batch_first=True
        )

        self.fc = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.policy_head = nn.Linear(hidden_dims[1], output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(state)  # [batch_size, seq_len, hidden_size]

        batch_size = gru_out.size(0)
        lengths = mask.sum(dim=1) - 1
        last_hidden = gru_out[torch.arange(batch_size), lengths]

        hidden = self.relu(self.fc(last_hidden))
        hidden = self.dropout(hidden)
        action = self.tanh(self.policy_head(hidden))
        return action


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ):
        super(CriticNetwork, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dims[0], batch_first=True
        )

        self.fc = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.q_head = nn.Linear(hidden_dims[1], 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        gru_out, _ = self.gru(state)

        batch_size = gru_out.size(0)
        lengths = mask.sum(dim=1) - 1
        last_hidden = gru_out[torch.arange(batch_size), lengths]

        combined = torch.cat([last_hidden, action], dim=1)
        hidden = self.relu(self.fc(combined))
        hidden = self.dropout(hidden)
        q_value = self.q_head(hidden)
        return q_value
