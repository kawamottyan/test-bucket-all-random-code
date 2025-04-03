import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    import os
    import numpy as np
    from typing import List, Dict, Callable
    import torch
    import pandas as pd
    return (
        Callable,
        DataLoader,
        Dataset,
        Dict,
        List,
        np,
        os,
        pd,
        pickle,
        torch,
        train_test_split,
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
def _(
    DataLoader,
    UserDataset,
    args_mut,
    batch_size,
    chain,
    frame_size,
    get_irsu,
    kwargs,
    np,
    num_items,
    num_workers,
    pd,
    pickle,
    self,
    torch,
    train_test_split,
    user_dict,
    utils,
):
    class FrameEnv():
        def __init__(self):
            self.train_user_dataset = None
            self.test_user_dataset = None
            self.embeddings = None
            self.key_to_id = None
            self.id_to_key = None
            self.frame_size = 10
            self.batch_size = 25
            self.num_workers = 1
            self.test_size = 0.05
            self._process_env()

            self.train_dataloader = DataLoader(
                self.train_user_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self.prepare_batch_wrapper,
            )

            self.test_dataloader = DataLoader(
                self.test_user_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self.prepare_batch_wrapper,
            )
        def _process_env(self):
            self._make_items_tensor(pickle.load(open('./data/ml-1m/ml20_pca128.pkl', "rb")))
            ratings = pd.read_csv('./data/ml-1m/ratings.csv', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
            print('ratings', ratings)
            self._prepare_dataset(ratings)

            train_users, test_users = train_test_split(self.users, test_size=self.test_size)
            train_users = self._sort_users_itemwise(user_dict, train_users)[2:]
            test_users = self._sort_users_itemwise(user_dict, test_users)
            self.base.train_user_dataset = UserDataset(train_users, user_dict)
            self.base.test_user_dataset = UserDataset(test_users, user_dict)

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
            print('self.key_to_id', self.key_to_id)
            print('self.id_to_key', self.id_to_key)
            print('self.embeddings', self.embeddings)
        
        def _prepare_dataset(self, ratings):
            def _try_progress_apply(dataframe, function):
                return dataframe.apply(function)
                
            df = ratings
        
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
            print('self.user_dict', self.user_dict)
            print('self.users', self.users)

        def _sort_users_itemwise(self, user_dict, users):
            return (
                pd.get()
                .Series(dict([(i, user_dict[i]["items"].shape[0]) for i in users]))
                .sort_values(ascending=False)
                .index
            )
        
        def _batch_contstate_discaction(
            batch, item_embeddings_tensor, frame_size, num_items, *args, **kwargs
        ):
            items_t, ratings_t, sizes_t, users_t = get_irsu(batch)
            items_emb = item_embeddings_tensor[items_t.long()]
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
            done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1
    
            one_hot_action = torch.zeros(b_size, num_items)
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
    
    
        def _truncate_dataset():
            num_items = kwargs.get("reduce_items_to")
            df = args_mut.df
    
            counts = df["movie_id"].value_counts().sort_values()
            to_remove = counts[:-num_items].index
            to_keep = counts[-num_items:].index
            to_keep_id = pd.get().Series(to_keep).apply(args_mut.base.key_to_id.get).values
            to_keep_mask = np.zeros(len(counts))
            to_keep_mask[to_keep_id] = 1
    
            args_mut.df = df.drop(df[df["movie_id"].isin(to_remove)].index)
    
            key_to_id_new = {}
            id_to_key_new = {}
            count = 0
    
            for idx, i in enumerate(list(args_mut.base.key_to_id.keys())):
                if i in to_keep:
                    key_to_id_new[i] = count
                    id_to_key_new[idx] = i
                    count += 1
    
            args_mut.base.embeddings = args_mut.base.embeddings[to_keep_mask]
            args_mut.base.key_to_id = key_to_id_new
            args_mut.base.id_to_key = id_to_key_new
    
            print(
                "action space is reduced to {} - {} = {}".format(
                    num_items + len(to_remove), len(to_remove), num_items
                )
            )
    
            return args_mut, kwargs
        
        def _build_data_pipeline():
            for call in chain:
                args_mut, _ = call(args_mut, kwargs)
            return args_mut, kwargs
        
        def embed_batch(batch, item_embeddings_tensor):
            return self._batch_contstate_discaction(batch, item_embeddings_tensor, frame_size=frame_size, num_items=num_items)
        
        def prepare_dataset(args_mut, kwargs):
            pipeline = [self._truncate_dataset, self._prepare_dataset]
            _build_data_pipeline(pipeline, kwargs, args_mut)

        def prepare_batch_wrapper(self, x):
            batch = utils.prepare_batch_static_size(
                x,
                self.base.embeddings,
                embed_batch=self.embed_batch,
                frame_size=self.frame_size,
            )
            return batch

        def train_batch(self):
            return next(iter(self.train_dataloader))

        def test_batch(self):
            return next(iter(self.test_dataloader))
    return (FrameEnv,)


@app.cell
def _():
    # dirs = DataPath(
    #     base="./data/ml-1m/",
    #     embeddings="ml20_pca128.pkl",
    #     ratings="ratings.csv",
    # )
    return


@app.cell
def _(FrameEnv):
    environment = FrameEnv()
    return (environment,)


@app.cell
def _(environment, tqdm):
    for batch in tqdm(environment.train_dataloader):
        print(batch)
    return (batch,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
