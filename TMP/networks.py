import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,  # ここは1182になる (1181 + 1)
        output_dim: int,  # ここは1181 (embedding_size)
        hidden_size1: int,
        hidden_size2: int,
        dropout_rate: float,
    ):
        super(ActorNetwork, self).__init__()
        # GRUレイヤーの追加
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size1, batch_first=True
        )

        self.dense2 = nn.Linear(hidden_size1, hidden_size2)
        self.policy_layer = nn.Linear(hidden_size2, output_dim)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # GRUの出力を取得
        gru_out, _ = self.gru(state)  # [batch_size, seq_len, hidden_size]

        # マスクを使用して最後の有効な出力を取得
        batch_size = gru_out.size(0)
        lengths = mask.sum(dim=1) - 1  # 各シーケンスの最後の有効なインデックス
        last_hidden = gru_out[torch.arange(batch_size), lengths]

        # 残りの処理は同じ
        hidden = self.relu(self.dense2(last_hidden))
        hidden = self.dropout(hidden)
        action = self.tanh(self.policy_layer(hidden))
        return action


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size1: int,
        hidden_size2: int,
        dropout_rate: float,
    ):
        super(CriticNetwork, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size1, batch_first=True
        )

        self.dense2 = nn.Linear(hidden_size1 + output_dim, hidden_size2)
        self.q_layer = nn.Linear(hidden_size2, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # GRUの出力を取得
        gru_out, _ = self.gru(state)

        # マスクを使用して最後の有効な出力を取得
        batch_size = gru_out.size(0)
        lengths = mask.sum(dim=1) - 1
        last_hidden = gru_out[torch.arange(batch_size), lengths]

        # アクションと結合して処理
        combined = torch.cat([last_hidden, action], dim=1)
        hidden = self.relu(self.dense2(combined))
        hidden = self.dropout(hidden)
        q_value = self.q_layer(hidden)
        return q_value
