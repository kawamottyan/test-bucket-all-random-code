import os
import tempfile
from datetime import datetime
from typing import Dict

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.common.models.networks import ActorNetwork, CriticNetwork
from src.common.utils.general import setup_logger

logger = setup_logger(__name__)
mlflow.set_tracking_uri("http://localhost:5000")


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value_loss = float("inf")
        self.early_stop = False

    def __call__(
        self,
        value_loss,
    ):
        if value_loss < self.best_value_loss - self.min_delta:
            self.best_value_loss = value_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %s out of %s", self.counter, self.patience
            )
            if self.counter >= self.patience:
                self.early_stop = True


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
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
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.es_check_frequency = config["early_stopping"]["check_frequency"]
        self.early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
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

                        if epoch_batch_count % self.es_check_frequency == 0:
                            test_value_loss, test_policy_loss = self.evaluate()

                            mlflow.log_metrics(
                                {
                                    "train_value_loss": train_value_loss,
                                    "train_policy_loss": train_policy_loss,
                                    "test_value_loss": test_value_loss,
                                    "test_policy_loss": test_policy_loss,
                                },
                                step=total_batch_count,
                            )

                            logger.info(
                                "Epoch %d, Batch %d - Current Value Loss: %.4f",
                                epoch,
                                epoch_batch_count,
                                train_value_loss,
                            )

                            self.early_stopping(test_value_loss)

                            if test_value_loss < best_value_loss:
                                best_value_loss = test_value_loss
                                logger.info(
                                    "New best value loss: %.4f", best_value_loss
                                )

                                self.save_best_model(
                                    epoch,
                                    epoch_batch_count,
                                    total_batch_count,
                                    best_value_loss,
                                )

                            if self.early_stopping.early_stop:
                                logger.info(
                                    "Early stopping triggered at epoch %d, batch %d",
                                    epoch,
                                    epoch_batch_count,
                                )
                                break

                    if self.early_stopping.early_stop:
                        break

                if self.early_stopping.early_stop:
                    break

            logger.info("Training completed - Best value loss: %.4f", best_value_loss)
            return best_value_loss

        except Exception as e:
            logger.error("Error during training: %s", str(e))
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
            logger.error("Error during evaluation: %s", str(e))
            raise

    def save_best_model(self, epoch, batch_count, total_batch_count, value_loss):
        logger.info("Saving new best model with value loss: %.4f", value_loss)
        self.actor.eval()
        self.critic.eval()

        try:
            example_batch = next(iter(self.train_loader))
            example_state = example_batch["state"][0:1].to(self.device)
            example_mask = example_batch["mask"][0:1].to(self.device)
            with torch.no_grad():
                actor_traced = torch.jit.trace_module(
                    self.actor, {"forward": (example_state, example_mask)}
                )
            with torch.no_grad():
                example_action = actor_traced.forward(example_state, example_mask)
            with torch.no_grad():
                critic_traced = torch.jit.trace_module(
                    self.critic,
                    {"forward": (example_state, example_action, example_mask)},
                )
            with mlflow.start_run(run_name="best_model", nested=True) as _:
                mlflow.log_params(
                    {
                        "value_loss": float(value_loss),
                        "epoch": epoch,
                        "batch": batch_count,
                        "total_batches": total_batch_count,
                    }
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    actor_path = os.path.join(tmp_dir, "actor.pt")
                    torch.jit.save(actor_traced, actor_path)
                    mlflow.log_artifact(actor_path, "best_actor_traced")
                    logger.info(
                        "Actor traced model saved to %s and logged to MLflow",
                        actor_path,
                    )

                    critic_path = os.path.join(tmp_dir, "critic.pt")
                    torch.jit.save(critic_traced, critic_path)
                    mlflow.log_artifact(critic_path, "best_critic_traced")
                    logger.info(
                        "Critic traced model saved to %s and logged to MLflow",
                        critic_path,
                    )

        except Exception as e:
            logger.error("Error saving best model: %s", str(e))
            return None, None
        finally:
            self.actor.train()
            self.critic.train()
            logger.info("Finished save_best_model method")
