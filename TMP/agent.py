from io import BytesIO
from datetime import datetime
import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.recommendation.common.storage.s3_handler import S3Handler
from src.models.networks import ActorNetwork, CriticNetwork
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(
        self, patience=7, min_delta=0.0001, verbose=True, path="checkpoint.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_value_loss = float("inf")
        self.early_stop = False

    def __call__(
        self,
        value_loss,
        actor_model,
        critic_model,
        target_actor_model,
        target_critic_model,
    ):
        if value_loss < self.best_value_loss - self.min_delta:
            self.save_checkpoint(
                value_loss,
                actor_model,
                critic_model,
                target_actor_model,
                target_critic_model,
            )
            self.best_value_loss = value_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(
        self,
        value_loss,
        actor_model,
        critic_model,
        target_actor_model,
        target_critic_model,
    ):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.best_value_loss:.6f} --> {value_loss:.6f}). Saving models..."
            )
        torch.save(
            {
                "actor_state_dict": actor_model.state_dict(),
                "critic_state_dict": critic_model.state_dict(),
                "target_actor_state_dict": target_actor_model.state_dict(),
                "target_critic_state_dict": target_critic_model.state_dict(),
                "value_loss": value_loss,
            },
            self.path,
        )


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = np.random.choice(self.buffer, batch_size, replace=False)
        state, action, reward, next_state, mask = zip(*batch)
        return (
            torch.stack(state),
            torch.stack(action),
            torch.stack(reward),
            torch.stack(next_state),
            torch.stack(mask),
        )


class DDPGAgent:
    def __init__(
        self,
        config,
        train_loader,
        eval_loader=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.input_dim = config["embedding_size"] + 1
        self.output_dim = config["embedding_size"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.action_noise = config["action_noise"]
        self.frame_size = config["frame_size"]
        self.train_sample_size = config["train_sample_size"]
        self.test_ratio = config["test_ratio"]
        self.test_item_size = config["test_item_size"]
        self.model_name = config["model_name"]
        self.results_path_name = config["result_dir_name"]
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.early_stopping_flag = config["early_stopping_flag"]
        if self.early_stopping_flag:
            self.early_stopping = EarlyStopping()
        self._initialize_networks(config)
        self._initialize_optimizers(config)
        self._soft_target_update()

    def _initialize_networks(self, config: Dict):
        self.actor = ActorNetwork(
            self.input_dim,
            self.output_dim,
            hidden_size1=config["actor_hidden_size1"],
            hidden_size2=config["actor_hidden_size2"],
            dropout_rate=config["actor_dropout_rate"],
        ).to(self.device)

        self.critic = CriticNetwork(
            self.input_dim,
            self.output_dim,
            hidden_size1=config["critic_hidden_size1"],
            hidden_size2=config["critic_hidden_size2"],
            dropout_rate=config["critic_dropout_rate"],
        ).to(self.device)

        self.target_actor = ActorNetwork(
            self.input_dim,
            self.output_dim,
            hidden_size1=config["actor_hidden_size1"],
            hidden_size2=config["actor_hidden_size2"],
            dropout_rate=config["actor_dropout_rate"],
        ).to(self.device)

        self.target_critic = CriticNetwork(
            self.input_dim,
            self.output_dim,
            hidden_size1=config["critic_hidden_size1"],
            hidden_size2=config["critic_hidden_size2"],
            dropout_rate=config["critic_dropout_rate"],
        ).to(self.device)

        self.replay_buffer = ReplayBuffer(capacity=100000)

    def _initialize_optimizers(self, config: Dict):
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config["actor_learning_rate"],
            weight_decay=config["actor_weight_decay"],
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config["critic_learning_rate"],
            weight_decay=config["critic_weight_decay"],
        )

    def _soft_target_update(self):
        with torch.no_grad():
            for target_param, param in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def reset_agent(self, config: Dict):
        self._initialize_networks(config)
        self._initialize_optimizers(config)

    def train(self, num_epochs: int):
        logger.info(
            "Starting training with the following parameters:\n"
            "Input Dimension: %d, Output Dimension: %d\n"
            "Actor Hidden Layer 1 Sizes: %d, Actor Hidden Layer 2 Sizes: %d\n"
            "Actor Dropout Rate: %.2f\n"
            "Critic Hidden Layer 1 Sizes: %d, Critic Hidden Layer 2 Sizes: %d\n"
            "Critic Dropout Rate: %.2f\n"
            "Actor Learning Rate: %.6f, Actor Weight Decay: %.6f\n"
            "Critic Learning Rate: %.6f, Critic Weight Decay: %.6f\n"
            "Gamma: %.3f, Tau: %.6f, Action Noise: %.2f\n"
            "Early Stopping: %s\n"
            "Device: %s, Number of Epochs: %d",
            self.input_dim,
            self.output_dim,
            self.actor.hidden_size1,
            self.actor.hidden_size2,
            self.actor.dropout_rate,
            self.critic.hidden_size1,
            self.critic.hidden_size2,
            self.critic.dropout_rate,
            self.actor_optimizer.defaults["lr"],
            self.actor_optimizer.defaults["weight_decay"],
            self.critic_optimizer.defaults["lr"],
            self.critic_optimizer.defaults["weight_decay"],
            self.gamma,
            self.tau,
            self.action_noise,
            "Enabled" if self.early_stopping_flag else "Disabled",
            self.device,
            num_epochs,
        )

        best_value_loss = float("inf")
        for epoch in range(1, num_epochs + 1):
            train_value_loss = 0.0
            train_policy_loss = 0.0
            batch_count = 0

            for batch in tqdm(self.train_loader):
                # モデルを学習モードに設定
                self.actor.train()
                self.critic.train()
                self.target_actor.train()
                self.target_critic.train()

                # バッチデータの取得と前処理
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # ReplayBufferには元のCPUデータを保存
                self.replay_buffer.push(
                    batch["state"],
                    batch["action"],
                    batch["reward"],
                    batch["next_state"],
                    batch["mask"],
                )
                # Replay Bufferからサンプリング
                if len(self.replay_buffer.buffer) >= self.train_sample_size:
                    (
                        sampled_states,
                        sampled_actions,
                        sampled_rewards,
                        sampled_next_states,
                        sampled_mask,
                    ) = self.replay_buffer.sample(self.train_sample_size)

                    sampled_states = sampled_states.to(self.device)
                    sampled_actions = sampled_actions.to(self.device)
                    sampled_rewards = sampled_rewards.to(self.device)
                    sampled_next_states = sampled_next_states.to(self.device)
                    sampled_mask = sampled_mask.to(self.device)

                    # Criticの更新
                    self.critic_optimizer.zero_grad()
                    with torch.no_grad():
                        # マスクを使用してターゲットアクションを生成
                        target_actions = self.target_actor(
                            sampled_next_states, sampled_mask
                        )

                        # マスクを使用してターゲットQ値を計算
                        target_q_value = self.target_critic(
                            sampled_next_states, target_actions, sampled_mask
                        )

                        target = sampled_rewards + self.gamma * target_q_value

                    # 現在のQ値の計算（マスク使用）
                    q_value = self.critic(sampled_states, sampled_actions, sampled_mask)

                    critic_loss = nn.MSELoss()(q_value, target)
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # Actorの更新
                    self.actor_optimizer.zero_grad()
                    # マスクを使用してアクションを予測
                    predicted_action = self.actor(sampled_states, sampled_mask)

                    noise = (
                        torch.randn_like(predicted_action).to(self.device)
                        * self.action_noise
                    )
                    predicted_action = predicted_action + noise

                    # マスクを使用してポリシー損失を計算
                    policy_loss = -self.critic(
                        sampled_states, predicted_action, sampled_mask
                    ).mean()

                    policy_loss.backward()
                    self.actor_optimizer.step()

                    self._soft_target_update()

                    train_value_loss += critic_loss.item()
                    train_policy_loss += policy_loss.item()
                    batch_count += 1

            # エポックごとの平均損失を計算
            if batch_count > 0:
                avg_train_value_loss = train_value_loss / batch_count
                avg_train_policy_loss = train_policy_loss / batch_count

                # 評価
                test_value_loss, test_policy_loss = self.evaluate()

                # Early stopping判定
                if self.early_stopping_flag:
                    self.early_stopping(
                        test_value_loss,
                        self.actor,
                        self.critic,
                        self.target_actor,
                        self.target_critic,
                    )

                    if self.early_stopping.early_stop:
                        print("Early stopping triggered")
                        # 最良のモデルを読み込む
                        best_value_loss = self.early_stopping.load_checkpoint(
                            self.actor,
                            self.critic,
                            self.target_actor,
                            self.target_critic,
                        )
                        break

                # ログ出力
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Value Loss: {avg_train_value_loss:.6f}, "
                    f"Train Policy Loss: {avg_train_policy_loss:.6f}, "
                    f"Test Value Loss: {test_value_loss:.6f}, "
                    f"Test Policy Loss: {test_policy_loss:.6f}"
                )

        return best_value_loss

    def evaluate(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        test_value_loss = 0.0
        test_policy_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating Batches"):
                states = batch["state"].to(self.device)
                next_states = batch["next_state"].to(self.device)
                actions = batch["action"].to(self.device)
                rewards = batch["reward"].unsqueeze(1).to(self.device)
                dones = batch["done"].unsqueeze(1).to(self.device)
                mask = batch["mask"].to(self.device)

                target_actions = self.target_actor(next_states, mask).to(self.device)
                target_q_value = self.target_critic(
                    next_states, target_actions, mask
                ).to(self.device)
                target = (rewards + (1.0 - dones) * self.gamma * target_q_value).to(
                    self.device
                )

                q_value = self.critic(states, actions, mask).to(self.device)
                value_loss = nn.MSELoss()(q_value, target)
                test_value_loss += value_loss.item()

                predicted_action = self.actor(states, mask).to(self.device)
                policy_loss = (
                    -self.critic(states, predicted_action, mask).mean().to(self.device)
                )
                test_policy_loss += policy_loss.item()
                batch_count += 1

        test_value_loss /= batch_count
        test_policy_loss /= batch_count

        return test_value_loss, test_policy_loss

    def save_model(
        self,
        s3_handler: S3Handler,
        bucket_name: str,
        actor_model_key: str,
        critic_model_key: str,
    ):
        self.actor.eval()
        self.critic.eval()

        example_input_actor = torch.randn(1, self.actor.input_dim).to(self.device)
        example_input_critic_state = torch.randn(1, self.critic.input_dim).to(
            self.device
        )
        example_input_critic_action = torch.randn(1, self.critic.output_dim).to(
            self.device
        )

        actor_traced = torch.jit.trace(self.actor, example_input_actor)
        critic_traced = torch.jit.trace(
            self.critic, (example_input_critic_state, example_input_critic_action)
        )

        actor_buffer = BytesIO()
        torch.jit.save(actor_traced, actor_buffer)
        actor_buffer.seek(0)
        s3_handler.s3_client.put_object(
            Bucket=bucket_name, Key=actor_model_key, Body=actor_buffer.getvalue()
        )
        logger.info("JIT Actor model saved to s3://%s/%s", bucket_name, actor_model_key)

        critic_buffer = BytesIO()
        torch.jit.save(critic_traced, critic_buffer)
        critic_buffer.seek(0)
        s3_handler.s3_client.put_object(
            Bucket=bucket_name, Key=critic_model_key, Body=critic_buffer.getvalue()
        )
        logger.info(
            "JIT Critic model saved to s3://%s/%s", bucket_name, critic_model_key
        )

    def load_model(
        self,
        s3_handler: S3Handler,
        bucket_name: str,
        actor_model_key: str,
        critic_model_key: str,
    ):
        try:
            actor_obj = s3_handler.get_s3_object(bucket_name, actor_model_key)
            actor_buffer = BytesIO(actor_obj["Body"].read())
            self.actor = torch.jit.load(actor_buffer).to(self.device)
            logger.info(
                "JIT Actor model loaded from s3://%s/%s",
                bucket_name,
                actor_model_key,
            )
        except Exception as e:
            logger.error("Failed to load actor model from S3: %s", e)
            raise e

        try:
            critic_obj = s3_handler.get_s3_object(bucket_name, critic_model_key)
            critic_buffer = BytesIO(critic_obj["Body"].read())
            self.critic = torch.jit.load(critic_buffer).to(self.device)
            logger.info(
                "JIT Critic model loaded from s3://%s/%s",
                bucket_name,
                critic_model_key,
            )
        except Exception as e:
            logger.error("Failed to load critic model from S3: %s", e)
            raise e

        logger.info("Actor Model Architecture and Parameters:")
        for name, param in self.actor.named_parameters():
            logger.info("Parameter: %s, Shape: %s", name, param.shape)

        logger.info("Critic Model Architecture and Parameters:")
        for name, param in self.critic.named_parameters():
            logger.info("Parameter: %s, Shape: %s", name, param.shape)
