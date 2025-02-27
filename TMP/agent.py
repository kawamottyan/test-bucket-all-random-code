from io import BytesIO
from datetime import datetime
import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.common.storage.s3_handler import S3Handler
from src.common.models.networks import ActorNetwork, CriticNetwork
import numpy as np
import mlflow
import mlflow.pytorch


logger = logging.getLogger(__name__)

MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
AWS_ACCESS_KEY_ID = "minioadmin"
AWS_SECRET_ACCESS_KEY = "minioadmin"
mlflow.set_tracking_uri("http://localhost:5000")


class EarlyStopping:
    def __init__(self, patience, min_delta, path):
        self.patience = patience
        self.min_delta = min_delta
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
        self.buffer[self.position] = (
            state,
            action,
            reward,
            next_state,
            mask,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        state, action, reward, next_state, mask = zip(*batch)

        return (
            torch.vstack(state),
            torch.vstack(action),
            torch.cat(reward),
            torch.vstack(next_state),
            torch.vstack(mask),
        )


class DDPGAgent:
    def __init__(
        self,
        config,
        train_loader,
        eval_loader=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.gamma = config["model"]["gamma"]
        self.tau = config["model"]["tau"]
        self.action_noise = config["model"]["action_noise"]
        self.train_sample_size = config["train"]["sample_size"]
        self.model_name = config["model"]["name"]
        self.results_path_name = config["dir"]["model"]
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.es_check_frequency = config["early_stopping"]["check_frequency"]
        self.early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
            path=config["early_stopping"]["path"],
        )
        self._initialize_networks(config)
        self._initialize_optimizers(config)
        self._soft_target_update()
        self.experiment_name = f"{self.model_name}_{self.created_at}"

    def _initialize_networks(self, config: Dict):
        self.actor = ActorNetwork(
            input_dim=config["data"]["embedding_dim"] + 1,
            output_dim=config["data"]["embedding_dim"],
            hidden_dims=[
                config["model"]["actor"]["hidden_dims"][0],
                config["model"]["actor"]["hidden_dims"][1],
            ],
            dropout_rate=config["model"]["actor"]["dropout_rate"],
        ).to(self.device)

        self.critic = CriticNetwork(
            input_dim=config["data"]["embedding_dim"] + 1,
            action_dim=config["data"]["embedding_dim"],
            hidden_dims=[
                config["model"]["critic"]["hidden_dims"][0],
                config["model"]["critic"]["hidden_dims"][1],
            ],
            dropout_rate=config["model"]["critic"]["dropout_rate"],
        ).to(self.device)

        self.target_actor = ActorNetwork(
            input_dim=config["data"]["embedding_dim"] + 1,
            output_dim=config["data"]["embedding_dim"],
            hidden_dims=[
                config["model"]["actor"]["hidden_dims"][0],
                config["model"]["actor"]["hidden_dims"][1],
            ],
            dropout_rate=config["model"]["actor"]["dropout_rate"],
        ).to(self.device)

        self.target_critic = CriticNetwork(
            input_dim=config["data"]["embedding_dim"] + 1,
            action_dim=config["data"]["embedding_dim"],
            hidden_dims=[
                config["model"]["critic"]["hidden_dims"][0],
                config["model"]["critic"]["hidden_dims"][1],
            ],
            dropout_rate=config["model"]["critic"]["dropout_rate"],
        ).to(self.device)

        self.replay_buffer = ReplayBuffer(config["model"]["buffer_capacity"])

    def _initialize_optimizers(self, config: Dict):
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config["model"]["actor"]["lr"],
            weight_decay=config["model"]["actor"]["weight_decay"],
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config["model"]["critic"]["lr"],
            weight_decay=config["model"]["critic"]["weight_decay"],
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
        try:
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run() as _:
                mlflow.log_params(
                    {
                        "gamma": self.gamma,
                        "tau": self.tau,
                        "action_noise": self.action_noise,
                        "train_sample_size": self.train_sample_size,
                        "actor_lr": self.actor_optimizer.defaults["lr"],
                        "critic_lr": self.critic_optimizer.defaults["lr"],
                        "num_epochs": num_epochs,
                        "device": str(self.device),
                        "check_frequency": self.es_check_frequency,
                    }
                )

            logger.info(
                "Starting DDPG training on %s for %d epochs with check frequency %d",
                self.device,
                num_epochs,
                self.es_check_frequency,
            )

            best_value_loss = float("inf")
            total_batch_count = 0

            for epoch in range(1, num_epochs + 1):
                self.actor.train()
                self.critic.train()
                self.target_actor.train()
                self.target_critic.train()

                train_value_loss = 0.0
                train_policy_loss = 0.0
                epoch_batch_count = 0

                for batch in tqdm(
                    self.train_loader, desc=f"Epoch {epoch}/{num_epochs}"
                ):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    with torch.no_grad():
                        actions = self.actor(batch["state"], batch["mask"])
                        noise = (
                            torch.randn_like(actions).to(self.device)
                            * self.action_noise
                        )
                        noisy_actions = actions + noise

                    self.replay_buffer.push(
                        batch["state"],
                        noisy_actions,
                        batch["reward"],
                        batch["next_state"],
                        batch["mask"],
                    )

                    if len(self.replay_buffer.buffer) >= self.train_sample_size:
                        states, actions, rewards, next_states, masks = (
                            self.replay_buffer.sample(self.train_sample_size)
                        )
                        states = states.to(self.device)
                        actions = actions.to(self.device)
                        rewards = rewards.to(self.device)
                        next_states = next_states.to(self.device)
                        masks = masks.to(self.device)

                        self.critic_optimizer.zero_grad()
                        with torch.no_grad():
                            target_actions = self.target_actor(next_states, masks)
                            target_q_value = self.target_critic(
                                next_states, target_actions, masks
                            )
                            rewards = rewards.unsqueeze(-1)
                            target = rewards + self.gamma * target_q_value

                        q_value = self.critic(states, actions, masks)
                        critic_loss = nn.MSELoss()(q_value, target)
                        critic_loss.backward()
                        self.critic_optimizer.step()

                        self.actor_optimizer.zero_grad()
                        predicted_action = self.actor(states, masks)
                        policy_loss = -self.critic(
                            states, predicted_action, masks
                        ).mean()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.parameters(), max_norm=1.0
                        )
                        self.actor_optimizer.step()

                        self._soft_target_update()

                        train_value_loss += critic_loss.item()
                        train_policy_loss += policy_loss.item()
                        epoch_batch_count += 1
                        total_batch_count += 1

                        if (
                            self.es_check_frequency > 0
                            and epoch_batch_count % self.es_check_frequency == 0
                        ):
                            test_value_loss, test_policy_loss = self.evaluate()

                            mlflow.log_metrics(
                                {
                                    "batch_value_loss": train_value_loss,
                                    "batch_policy_loss": train_policy_loss,
                                    "test_value_loss": test_value_loss,
                                    "test_policy_loss": test_policy_loss,
                                },
                                step=total_batch_count,
                            )

                            logger.info(
                                f"Epoch {epoch}, Batch {epoch_batch_count} - "
                                f"Current Value Loss: {train_value_loss:.4f}"
                            )

                            self.early_stopping(
                                test_value_loss,
                                self.actor,
                                self.critic,
                                self.target_actor,
                                self.target_critic,
                            )

                            if test_value_loss < best_value_loss:
                                best_value_loss = test_value_loss
                                logger.info(
                                    f"New best value loss: {best_value_loss:.4f}"
                                )
                                mlflow.pytorch.log_model(self.actor, "best_actor_model")
                                mlflow.pytorch.log_model(
                                    self.critic, "best_critic_model"
                                )

                            if self.early_stopping.early_stop:
                                logger.info(
                                    f"Early stopping triggered at epoch {epoch}, batch {epoch_batch_count}"
                                )
                                break

                    if self.early_stopping.early_stop:
                        break

                if self.early_stopping.early_stop:
                    break

            logger.info(f"Training completed - Best value loss: {best_value_loss:.4f}")
            return best_value_loss

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        test_value_loss = 0.0
        test_policy_loss = 0.0
        batch_count = 0

        try:
            with torch.no_grad():
                for batch in tqdm(self.eval_loader, desc="Evaluating"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    states = batch["state"]
                    next_states = batch["next_state"]
                    actions = batch["action"]
                    rewards = batch["reward"].unsqueeze(1)
                    mask = batch["mask"]

                    target_actions = self.target_actor(next_states, mask)
                    target_q_value = self.target_critic(
                        next_states, target_actions, mask
                    )
                    target = rewards + self.gamma * target_q_value

                    q_value = self.critic(states, actions, mask)
                    value_loss = nn.MSELoss()(q_value, target)
                    test_value_loss += value_loss.item()

                    predicted_action = self.actor(states, mask)
                    policy_loss = -self.critic(states, predicted_action, mask).mean()
                    test_policy_loss += policy_loss.item()

                    batch_count += 1

            if batch_count > 0:
                test_value_loss /= batch_count
                test_policy_loss /= batch_count

            return test_value_loss, test_policy_loss

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

    def save_model(
        self,
        s3_handler: S3Handler,
        bucket_name: str,
        actor_model_key: str,
        critic_model_key: str,
    ):
        try:
            self.actor.eval()
            self.critic.eval()

            example_state = next(iter(self.train_loader))["state"][0:1].to(self.device)
            example_mask = next(iter(self.train_loader))["mask"][0:1].to(self.device)

            def actor_wrapper(state):
                return self.actor(state, example_mask)

            def critic_wrapper(state, action):
                return self.critic(state, action, example_mask)

            actor_traced = torch.jit.trace(actor_wrapper, example_state)

            with torch.no_grad():
                example_action = self.actor(example_state, example_mask)

            critic_traced = torch.jit.trace(
                critic_wrapper, (example_state, example_action)
            )

            actor_buffer = BytesIO()
            torch.jit.save(actor_traced, actor_buffer)
            actor_buffer.seek(0)
            s3_handler.s3_client.put_object(
                Bucket=bucket_name, Key=actor_model_key, Body=actor_buffer.getvalue()
            )
            logger.info(
                "JIT Actor model saved to s3://%s/%s", bucket_name, actor_model_key
            )

            critic_buffer = BytesIO()
            torch.jit.save(critic_traced, critic_buffer)
            critic_buffer.seek(0)
            s3_handler.s3_client.put_object(
                Bucket=bucket_name, Key=critic_model_key, Body=critic_buffer.getvalue()
            )
            logger.info(
                "JIT Critic model saved to s3://%s/%s", bucket_name, critic_model_key
            )

        except Exception as e:
            logger.error(f"Error during model saving: {str(e)}")
            raise

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

            critic_obj = s3_handler.get_s3_object(bucket_name, critic_model_key)
            critic_buffer = BytesIO(critic_obj["Body"].read())
            self.critic = torch.jit.load(critic_buffer).to(self.device)

            logger.info(f"Models loaded from s3://{bucket_name}")

        except Exception as e:
            logger.error(f"Failed to load models from S3: {str(e)}")
            raise
