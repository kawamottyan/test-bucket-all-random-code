import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    import os
    import numpy as np
    from typing import List, Dict, Callable
    import torch
    import pandas as pd
    import torch_optimizer as optim
    import gc
    import copy
    from tqdm import tqdm
    import torch.nn.functional as F
    from torch.distributions import Categorical
    import warnings
    return (
        Callable,
        Categorical,
        DataLoader,
        Dataset,
        Dict,
        F,
        List,
        copy,
        gc,
        nn,
        np,
        optim,
        os,
        pd,
        pickle,
        torch,
        tqdm,
        train_test_split,
        warnings,
    )


@app.cell
def _(Dataset):
    class UserDataset(Dataset):
        def __init__(self, users, user_dict):
            self.users = users
            self.user_dict = user_dict

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            idx = self.users[idx]
            group = self.user_dict[idx]
            items = group["items"][:]
            rates = group["ratings"][:]
            size = items.shape[0]

            return {"items": items, "rates": rates, "sizes": size, "users": idx}
    return (UserDataset,)


@app.cell
def _(DataLoader, UserDataset, np, pd, pickle, torch, train_test_split):
    class FrameEnv():
        def __init__(self):
            self.train_user_dataset = None
            self.test_user_dataset = None
            self.embeddings = None
            self.key_to_id = None
            self.id_to_key = None
            self.valid_movie_ids = set() 
            self.frame_size = 10
            self.batch_size = 25
            self.num_workers = 0
            self.test_size = 0.05
            self.num_items = 0
            self._process_env()



            self.train_dataloader = DataLoader(
                self.train_user_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.prepare_batch_wrapper,
            )

            self.test_dataloader = DataLoader(
                self.test_user_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.prepare_batch_wrapper,
            )
        def _process_env(self):
            self._make_items_tensor(pickle.load(open('./data/ml-1m/ml20_pca128.pkl', "rb")))
            ratings = pd.read_csv('./data/ml-1m/ratings.csv', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
            self._prepare_dataset(ratings)

            train_users, test_users = train_test_split(self.users, test_size=self.test_size)
            self.train_user_dataset = UserDataset(train_users[2:], self.user_dict)
            self.test_user_dataset = UserDataset(test_users, self.user_dict)

        def _make_items_tensor(self, items_embeddings_key_dict):
            keys = list(sorted(items_embeddings_key_dict.keys()))
            self.key_to_id = dict(zip(keys, range(len(keys))))
            self.id_to_key = dict(zip(range(len(keys)), keys))
            items_embeddings_id_dict = {}
            for k in items_embeddings_key_dict.keys():
                items_embeddings_id_dict[self.key_to_id[k]] = items_embeddings_key_dict[k]
            self.embeddings = torch.stack(
                [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
            )

        def _prepare_dataset(self, ratings):
            def _try_progress_apply(dataframe, function):
                return dataframe.apply(function)

            df = ratings
            self.valid_movie_ids = set(self.key_to_id.keys())
            df = df[df['movie_id'].isin(self.valid_movie_ids)].copy()
            self.num_items = len(self.valid_movie_ids)

            df["rating"] = _try_progress_apply(df["rating"], lambda i: 2 * (i - 2.5))
            df["movie_id"] = _try_progress_apply(df["movie_id"], self.key_to_id.get)
            users = df[["user_id", "movie_id"]].groupby(["user_id"]).size()
            users = users[users > self.frame_size].sort_values(ascending=False).index
            ratings = (
                df.sort_values(by="timestamp")
                .set_index("user_id")
                .drop("timestamp", axis=1)
                .groupby("user_id")
            )

            user_dict = {}

            def app(x):
                user_id = x.index[0]
                user_dict[user_id] = {}
                user_dict[user_id]["items"] = x["movie_id"].values
                user_dict[user_id]["ratings"] = x["rating"].values

            _try_progress_apply(ratings, app)

            self.user_dict = user_dict
            self.users = users

        def prepare_batch_wrapper(self, x):
            return self._prepare_batch_static_size(x)

        def _prepare_batch_static_size(self, batch):

            def rolling_window(a, window):
                shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
                strides = a.strides + (a.strides[-1],)
                return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

            item_t, ratings_t, sizes_t, users_t = [], [], [], []
            for i in range(len(batch)):
                item_t.append(batch[i]["items"])
                ratings_t.append(batch[i]["rates"])
                sizes_t.append(batch[i]["sizes"])
                users_t.append(batch[i]["users"])

            item_t = np.concatenate([rolling_window(i, self.frame_size + 1) for i in item_t], 0)
            ratings_t = np.concatenate(
                [rolling_window(i, self.frame_size + 1) for i in ratings_t], 0
            )

            item_t = torch.tensor(item_t)
            users_t = torch.tensor(users_t)
            ratings_t = torch.tensor(ratings_t).float()
            sizes_t = torch.tensor(sizes_t)

            batch = {"items": item_t, "users": users_t, "ratings": ratings_t, "sizes": sizes_t}

            return self._embed_batch(batch)

        def _embed_batch(self, batch):
            return self._batch_contstate_discaction(batch)

        def _batch_contstate_discaction(self, batch):
            def get_irsu(batch):
                items_t, ratings_t, sizes_t, users_t = (
                    batch["items"],
                    batch["ratings"],
                    batch["sizes"],
                    batch["users"],
                )
                return items_t, ratings_t, sizes_t, users_t

            items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
            items_emb = self.embeddings[items_t.long()]
            b_size = ratings_t.size(0)

            items = items_emb[:, :-1, :].view(b_size, -1)
            next_items = items_emb[:, 1:, :].view(b_size, -1)
            ratings = ratings_t[:, :-1]
            next_ratings = ratings_t[:, 1:]

            state = torch.cat([items, ratings], 1)
            next_state = torch.cat([next_items, next_ratings], 1)
            action = items_t[:, -1]
            reward = ratings_t[:, -1]

            done = torch.zeros(b_size)
            done[torch.cumsum(sizes_t - self.frame_size, dim=0) - 1] = 1

            one_hot_action = torch.zeros(b_size, self.num_items)
            one_hot_action.scatter_(1, action.view(-1, 1), 1)

            batch = {
                "state": state,
                "action": one_hot_action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "meta": {"users": users_t, "sizes": sizes_t},
            }
            return batch
    return (FrameEnv,)


@app.cell
def _(FrameEnv):
    environment = FrameEnv()
    return (environment,)


@app.cell
def _(environment, torch):
    num_items = environment.num_items
    input = 1290
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, input, num_items


@app.cell
def _(input, nn, num_items, optim):
    class Beta(nn.Module):
        def __init__(self):
            super().__init__() 
            self.net = nn.Sequential(
                nn.Linear(input, num_items),
                nn.Softmax()
            )
            self.optim = optim.RAdam(self.net.parameters(), lr=1e-5, weight_decay=1e-5)
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, state, action):

            predicted_action = self.net(state)

            loss = self.criterion(predicted_action, action.argmax(1))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            return predicted_action.detach()
    return (Beta,)


@app.cell
def _(F, nn, torch):
    class Critic(nn.Module):
        def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-5):
            super().__init__() 

            self.drop_layer = nn.Dropout(p=0.5)

            self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, 1)

            self.linear3.weight.data.uniform_(-init_w, init_w)
            self.linear3.bias.data.uniform_(-init_w, init_w)

        def forward(self, state, action):
            value = torch.cat([state, action], 1)
            value = F.relu(self.linear1(value))
            value = self.drop_layer(value)
            value = F.relu(self.linear2(value))
            value = self.drop_layer(value)
            value = self.linear3(value)
            return value
    return (Critic,)


@app.cell
def _(Categorical, F, nn, torch):
    class DiscreteActor(nn.Module):
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


        def _select_action_with_TopK_correction(
            self, state, beta, action, K):
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
    return (DiscreteActor,)


@app.cell
def _(Beta, device):
    beta_net   = Beta().to(device)
    return (beta_net,)


@app.cell
def _(Critic, DiscreteActor, device, input, num_items):
    value_net  = Critic(input, num_items, 2048, 54e-2).to(device)
    policy_net = DiscreteActor(input, num_items, 2048).to(device)
    return policy_net, value_net


@app.cell
def _(np, torch):
    class ChooseREINFORCE:
        def __init__(self):
            self.method = ChooseREINFORCE.reinforce_with_TopK_correction

        @staticmethod
        def reinforce_with_TopK_correction(policy, returns):
            scalar_policy_loss_terms = []
            try:
                num_items = len(policy.lambda_k)
                min_len = min(num_items, len(policy.correction), len(policy.saved_log_probs), len(returns))

                if min_len == 0:
                    return torch.tensor(0.0, requires_grad=True)

                target_device = returns.device

                for i in range(min_len):
                    l_k_vec = policy.lambda_k[i]
                    corr_vec = policy.correction[i]
                    log_prob_vec = policy.saved_log_probs[i]
                    R_scalar = returns[i]

                    if not isinstance(l_k_vec, torch.Tensor): l_k_vec = torch.tensor(l_k_vec, device=target_device, dtype=torch.float32)
                    if not isinstance(corr_vec, torch.Tensor): corr_vec = torch.tensor(corr_vec, device=target_device, dtype=torch.float32)
                    if not isinstance(log_prob_vec, torch.Tensor): log_prob_vec = torch.tensor(log_prob_vec, device=target_device, dtype=torch.float32)

                    l_k_vec = l_k_vec.to(target_device)
                    corr_vec = corr_vec.to(target_device)
                    log_prob_vec = log_prob_vec.to(target_device)


                    loss_term_vec = l_k_vec * corr_vec * -log_prob_vec * R_scalar
                    scalar_loss_term = loss_term_vec.sum()
                    scalar_policy_loss_terms.append(scalar_loss_term)

            except (AttributeError, TypeError, ValueError, IndexError, RuntimeError) as e:
                return torch.tensor(0.0, requires_grad=True)
            except Exception as e:
                 return torch.tensor(0.0, requires_grad=True)


            if not scalar_policy_loss_terms:
                return torch.tensor(0.0, requires_grad=True)

            try:
                 target_device = scalar_policy_loss_terms[0].device
                 scalar_policy_loss_terms = [ (t if isinstance(t, torch.Tensor) else torch.tensor(t, device=target_device)).to(target_device)
                                                for t in scalar_policy_loss_terms ]
                 total_policy_loss = torch.stack(scalar_policy_loss_terms).sum()
            except Exception:
                 total_policy_loss = torch.tensor(0.0, requires_grad=True)

            return total_policy_loss

        def __call__(self, policy, optimizer, learn=True):
            R = 0
            returns_list = []
            if hasattr(policy, 'rewards') and isinstance(policy.rewards, (list, tuple)):
                for r in policy.rewards[::-1]:
                    if isinstance(r, torch.Tensor):
                        r_val = r.item() if r.numel() == 1 else 0.0
                    elif isinstance(r, (int, float, np.number)):
                        r_val = float(r)
                    else:
                        r_val = 0.0
                    R = r_val + 0.99 * R
                    returns_list.insert(0, R)

            if not returns_list:
                if hasattr(policy, 'gc'): policy.gc()
                return torch.tensor(0.0)

            try:
                # Try to determine device from optimizer or policy parameters
                if isinstance(optimizer, torch.optim.Optimizer) and optimizer.param_groups:
                    target_device = optimizer.param_groups[0]['params'][0].device
                elif hasattr(policy, 'parameters'):
                     try:
                          target_device = next(policy.parameters()).device
                     except StopIteration:
                          target_device = torch.device("cpu") # Fallback
                else:
                     target_device = torch.device("cpu") # Default fallback

                returns = torch.tensor(returns_list, dtype=torch.float32, device=target_device)

                if returns.numel() > 1:
                     mean = returns.mean()
                     std = returns.std()
                     returns = (returns - mean) / (std + 0.0001)
                     if torch.isnan(returns).any() or torch.isinf(returns).any():
                         returns = torch.zeros_like(returns) # Replace NaN/Inf with zeros
                elif returns.numel() == 1:
                     returns = torch.tensor([0.0], dtype=torch.float32, device=target_device)
                else: # Empty
                     if hasattr(policy, 'gc'): policy.gc()
                     return torch.tensor(0.0)

            except Exception:
                 if hasattr(policy, 'gc'): policy.gc()
                 return torch.tensor(0.0)

            policy_loss = self.method(policy, returns)

            if not isinstance(policy_loss, torch.Tensor) or not policy_loss.requires_grad:
                 if hasattr(policy, 'gc'): policy.gc()
                 return policy_loss

            if learn:
                try:
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
                except Exception:
                     try:
                          optimizer.zero_grad()
                     except Exception:
                          pass # Ignore error in zero_grad if backward failed badly

            if hasattr(policy, 'gc'):
                policy.gc()

            return policy_loss
    return (ChooseREINFORCE,)


@app.cell
def _(ChooseREINFORCE, copy, optim, torch):
    class Reinforce():
        def __init__(self, policy_net, value_net, beta_net):
            super().__init__() 
            self.beta_net = beta_net

            target_policy_net = copy.deepcopy(policy_net)
            target_value_net = copy.deepcopy(value_net)

            target_policy_net.eval()
            target_value_net.eval()

            self._soft_update(value_net, target_value_net, soft_tau=1.0)
            self._soft_update(policy_net, target_policy_net, soft_tau=1.0)

            value_optimizer = optim.Ranger(
                value_net.parameters(), lr=1e-5, weight_decay=1e-2
            )
            policy_optimizer = optim.Ranger(
                policy_net.parameters(), lr=1e-5, weight_decay=1e-2
            )

            self.nets = {
                "value_net": value_net,
                "target_value_net": target_value_net,
                "policy_net": policy_net,
                "target_policy_net": target_policy_net,
            }

            self.optimizers = {
                "policy_optimizer": policy_optimizer,
                "value_optimizer": value_optimizer,
            }

            self.params = {
                "reinforce": ChooseREINFORCE(),
                "K": 10,
                "gamma": 0.99,
                "min_value": -10,
                "max_value": 10,
                "policy_step": 5,
                "soft_tau": 0.001,
            }

            self.loss_layout = {
                "test": {"value": [], "policy": [], "step": []},
                "train": {"value": [], "policy": [], "step": []},
            }

            self._step = 0

        def to(self, device):
            self.nets = {k: v.to(device) for k, v in self.nets.items()}
            self.device = device
            return self

        def step(self):
            self._step += 1

        def _soft_update(self, net, target_net, soft_tau=1e-2):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        def _get_base_batch(self, batch, device=torch.device("cpu"), done=True):
            b = [
                batch["state"],
                batch["action"],
                batch["reward"].unsqueeze(1),
                batch["next_state"],
            ]
            if done:
                b.append(batch["done"].unsqueeze(1))
            else:
                batch.append(torch.zeros_like(batch["reward"]))
            return [i.to(device) for i in b]

        def update(
            self,
            batch,
            device=torch.device("cpu"),
            debug=None,
            learn=True,
        ):

            state, action, reward, next_state, done = self._get_base_batch(batch)

            predicted_probs = self.nets["policy_net"].select_action(state=state, beta=self.beta_net.forward, action=action, K=self.params["K"])
            mx = predicted_probs.max(dim=1).values
            reward = self.nets["value_net"](state, predicted_probs).detach()
            self.nets["policy_net"].rewards.append(reward.mean())

            value_loss = self.value_update(
                batch,
                self.params,
                self.nets,
                self.optimizers,
                device=device,
                debug=debug,
                learn=True,
                step=self.step,
            )

            if self._step % self.params["policy_step"] == 0 and self._step > 0:
                policy_loss = self.params["reinforce"](
                    self.nets["policy_net"],
                    self.optimizers["policy_optimizer"],
                )

                self._soft_update(
                    self.nets["value_net"], self.nets["target_value_net"], soft_tau=self.params["soft_tau"]
                )
                self._soft_update(
                    self.nets["policy_net"], self.nets["target_policy_net"], soft_tau=self.params["soft_tau"]
                )

                losses = {
                    "value": value_loss.item(),
                    "policy": policy_loss.item(),
                    "step": self._step,
                }

                print('policy_loss', policy_loss)
                return policy_loss

        def value_update(
            self,
            batch,
            params,
            nets,
            optimizers,
            device=torch.device("cpu"),
            debug=None,
            learn=False,
            step=-1,
        ):
            def temporal_difference(reward, done, gamma, target):
                return reward + (1.0 - done) * gamma * target

            state, action, reward, next_state, done = self._get_base_batch(batch, device=device)

            with torch.no_grad():
                next_action = nets["target_policy_net"](next_state)
                target_value = nets["target_value_net"](next_state, next_action.detach())
                expected_value = temporal_difference(
                    reward, done, params["gamma"], target_value
                )
                expected_value = torch.clamp(
                    expected_value, params["min_value"], params["max_value"]
                )

            value = nets["value_net"](state, action)

            value_loss = torch.pow(value - expected_value.detach(), 2).mean()

            optimizers["value_optimizer"].zero_grad()
            value_loss.backward(retain_graph=True)
            optimizers["value_optimizer"].step()

            return value_loss
    return (Reinforce,)


@app.cell
def _(Reinforce, beta_net, device, policy_net, value_net):
    reinforce = Reinforce(policy_net, value_net, beta_net)
    reinforce = reinforce.to(device)
    return (reinforce,)


@app.cell
def _():
    n_epochs = 1
    return (n_epochs,)


@app.cell
def _(environment, n_epochs, reinforce, tqdm):
    for epoch in range(n_epochs):
        for batch in tqdm(environment.train_dataloader):
            reinforce.update(batch)
            reinforce.step()
    return batch, epoch


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
