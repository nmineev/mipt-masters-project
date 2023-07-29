import random
import pprint
import gc
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tqdm
import torch
from torchrl.modules import EGreedyWrapper
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule, TensorDictSequential
import dgl
from omegaconf import OmegaConf
from .environment import GCQNEnv
from .test_utilities import test_QValueModule
from .models import GCQN, GCQNQValueModule, TGQN, RNNQ, SVDQ, Random
from .train_valid_loops import train_loop, valid_loop


def get_dataset(dataset_name):
    if dataset_name == "steam":
        df = pd.read_csv("./data/steam.csv", dtype={
            "app_id": np.int32, "user_id": np.int32, "review_id": np.int32,
            "helpful": np.int16, "funny": np.int16, "hours": np.float16})
        df["timestamp"] = (pd.to_datetime(df.date) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        df["rating"] = df.is_recommended.astype(int) * 5
        df.sort_values("timestamp", inplace=True, ignore_index=True)
        df.rename(columns={"app_id": "item_id"}, inplace=True)
    elif dataset_name == "goodreads":
        df = pd.read_csv("./data/goodreads.csv")
        df.drop(columns=["has_spoiler", "review_id", "old_timestamp", "user_id_original", "item_id_original", "reward"],
                inplace=True)
        n_items = 1500
        df = df[df.item_id.isin(df.item_id.value_counts()[:n_items].index)].copy()
        df.sort_values("timestamp", inplace=True, ignore_index=True)
    else:
        df = pd.read_csv(
            "./data/movielens1m.dat",
            sep="::",
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        df.sort_values("timestamp", inplace=True, ignore_index=True)
    gc.collect()
    # Drop users with less than min_num_positive_ratings positive ratings
    min_num_positive_ratings = 28
    max_num_positive_ratings = 1000
    max_num_negative_ratings = 1000
    user_num_positive_ratings = df[df.rating > 3].user_id.value_counts()
    users_ids_with_min_num_positive_ratings = user_num_positive_ratings.index[
        user_num_positive_ratings >= min_num_positive_ratings]
    users_ids_with_max_num_positive_ratings = user_num_positive_ratings.index[
        user_num_positive_ratings > max_num_positive_ratings]
    user_num_negative_ratings = df[df.rating <= 3].user_id.value_counts()
    users_ids_with_max_num_negative_ratings = user_num_negative_ratings.index[
        user_num_negative_ratings > max_num_negative_ratings]
    df = df[df.user_id.isin(users_ids_with_min_num_positive_ratings)
            & ~df.user_id.isin(users_ids_with_max_num_positive_ratings)
            & ~df.user_id.isin(users_ids_with_max_num_negative_ratings)].copy()

    user_id_encoder = LabelEncoder().fit(df.user_id)
    item_id_encoder = LabelEncoder().fit(df.item_id)
    df["user_id_original"] = df.user_id.copy()
    df["item_id_original"] = df.item_id.copy()
    df["user_id"] = user_id_encoder.transform(df.user_id)
    df["item_id"] = item_id_encoder.transform(df.item_id)
    NUM_USERS = df.user_id.unique().shape[0]
    NUM_ITEMS = df.item_id.unique().shape[0]
    ITEM_ID_PAD = NUM_ITEMS
    df["reward"] = (df.rating > 3).astype(int)

    return df, NUM_USERS, NUM_ITEMS, ITEM_ID_PAD


def get_users_pos_items(df, NUM_USERS, ITEM_ID_PAD):
    max_num_pos_items = df[df.reward == 1].user_id.value_counts().max()
    users_pos_items = torch.full((NUM_USERS, max_num_pos_items), ITEM_ID_PAD)

    for user_id in tqdm.tqdm(range(NUM_USERS), desc="Create users_pos_items"):
        user_pos_items = df[(df.user_id == user_id) & (df.reward == 1)].item_id.values
        users_pos_items[user_id, :len(user_pos_items)] = torch.from_numpy(user_pos_items)
        #if user_id == 1500: break ##################
    return users_pos_items


def get_users_items_to_take_actions(df, NUM_USERS, ITEM_ID_PAD):
    num_items_to_take_action = 1000
    users_items_to_take_actions = torch.full((NUM_USERS, num_items_to_take_action), ITEM_ID_PAD)

    for user_id in tqdm.tqdm(range(NUM_USERS), desc="Create users_items_to_take_actions"):
        user_items_to_take_action = df[df.user_id == user_id].sort_values("reward", ascending=False)\
                                        .item_id.values[:num_items_to_take_action]
        users_items_to_take_actions[user_id, :len(user_items_to_take_action)] = torch.from_numpy(
            user_items_to_take_action)
        #if user_id == 1500: break #########################################3
    return users_items_to_take_actions


def get_interactions_graph(df, NUM_ITEMS):
    src = torch.cat([torch.from_numpy(df.user_id.values + NUM_ITEMS), torch.from_numpy(df.item_id.values)])
    dst = torch.cat([torch.from_numpy(df.item_id.values), torch.from_numpy(df.user_id.values + NUM_ITEMS)])
    interactions_graph = dgl.graph((src, dst))
    interactions_graph.edata["reward"] = torch.cat(
        [torch.from_numpy(df.reward.values), torch.from_numpy(df.reward.values)])
    return interactions_graph


# def get_pos_interactions_graph(df, NUM_ITEMS):
#     src = torch.cat([torch.from_numpy(df[df.reward == 1].user_id.values + NUM_ITEMS),
#                      torch.from_numpy(df[df.reward == 1].item_id.values)])
#     dst = torch.cat([torch.from_numpy(df[df.reward == 1].item_id.values),
#                      torch.from_numpy(df[df.reward == 1].user_id.values + NUM_ITEMS)])
#     pos_interactions_graph = dgl.graph((src, dst))
#     pos_interactions_graph.edata["reward"] = torch.cat(
#         [torch.from_numpy(df[df.reward == 1].reward.values), torch.from_numpy(df[df.reward == 1].reward.values)])
#     return pos_interactions_graph


def get_model(cfg, num_users, num_items, item_id_pad, df, interactions_graph, device):
    if cfg.common.model_type == "tgqn":
        memory_dim = cfg.tgqn.memory_dim
        embedding_dim = cfg.tgqn.embedding_dim
        reward_dim = cfg.tgqn.reward_dim
        num_heads = cfg.tgqn.num_heads
        num_layers = cfg.tgqn.num_layers
        num_neighbours = cfg.tgqn.num_neighbours
        gnn_type = cfg.tgqn.gnn_type
        rnn_type = cfg.tgqn.rnn_type
        memory_alpha = cfg.tgqn.memory_alpha
        use_users_raw_embeddings = cfg.tgqn.use_users_raw_embeddings
        items_memory_batch_agg = cfg.tgqn.items_memory_batch_agg
        items_embedding_module_input = cfg.tgqn.items_embedding_module_input
        users_embedding_module_input = cfg.tgqn.users_embedding_module_input
        predictor_type = cfg.tgqn.predictor_type
        predictor_hidden_dim = cfg.tgqn.predictor_hidden_dim
        users_predictor_input = cfg.tgqn.users_predictor_input
        items_predictor_input = cfg.tgqn.items_predictor_input
        use_items_memory_as_hidden = cfg.tgqn.use_items_memory_as_hidden
        pos_interactions_only = cfg.tgqn.pos_interactions_only
        dropout = cfg.tgqn.dropout

        tgqn = TensorDictModule(
            TGQN(num_users=num_users, num_items=num_items, interactions_graph=interactions_graph,
                 memory_dim=memory_dim, embedding_dim=embedding_dim, reward_dim=reward_dim,
                 num_heads=num_heads, num_layers=num_layers, num_neighbours=num_neighbours,
                 gnn_type=gnn_type, rnn_type=rnn_type, memory_alpha=memory_alpha,
                 use_users_raw_embeddings=use_users_raw_embeddings, items_memory_batch_agg=items_memory_batch_agg,
                 items_embedding_module_input=items_embedding_module_input, users_embedding_module_input=users_embedding_module_input,
                 predictor_type=predictor_type, predictor_hidden_dim=predictor_hidden_dim, users_predictor_input=users_predictor_input,
                 items_predictor_input=items_predictor_input, use_items_memory_as_hidden=use_items_memory_as_hidden,
                 pos_interactions_only=pos_interactions_only, dropout=dropout, device=device),
            in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
            out_keys=["action_value"])
        return tgqn

    elif cfg.common.model_type == "gcqn":
        raw_embedding_dim = cfg.gcqn.raw_embedding_dim
        gnn_embedding_dim = cfg.gcqn.gnn_embedding_dim
        rnn_embedding_dim = cfg.gcqn.rnn_embedding_dim
        reward_dim = cfg.gcqn.reward_dim
        num_heads = cfg.gcqn.num_heads
        num_layers = cfg.gcqn.num_layers
        num_neighbours = cfg.gcqn.num_neighbours
        gnn_type = cfg.gcqn.gnn_type
        rnn_type = cfg.gcqn.rnn_type
        predictor_type = cfg.gcqn.predictor_type
        use_rewards_in_rnn = cfg.gcqn.use_rewards_in_rnn
        predictor_hidden_dim = cfg.gcqn.predictor_hidden_dim

        gcqn = TensorDictModule(
            GCQN(num_users=num_users, num_items=num_items, interactions_graph=interactions_graph,
                 raw_embedding_dim=raw_embedding_dim, gnn_embedding_dim=gnn_embedding_dim, rnn_embedding_dim=rnn_embedding_dim,
                 reward_dim=reward_dim, num_heads=num_heads, num_layers=num_layers, num_neighbours=num_neighbours,
                 gnn_type=gnn_type, rnn_type=rnn_type, predictor_type=predictor_type, use_rewards_in_rnn=use_rewards_in_rnn,
                 predictor_hidden_dim=predictor_hidden_dim, device=device),
            in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
            out_keys=["action_value"])
        return gcqn

    elif cfg.common.model_type == "rnnq":
        embedding_dim = cfg.rnnq.embedding_dim
        rnn_type = cfg.rnnq.rnn_type
        reward_dim = cfg.rnnq.reward_dim
        use_rewards = cfg.rnnq.use_rewards

        rnnq = TensorDictModule(
            RNNQ(num_items=num_items, embedding_dim=embedding_dim, rnn_type=rnn_type, reward_dim=reward_dim,
                 use_rewards=use_rewards,),
            in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
            out_keys=["action_value"])
        return rnnq

    elif cfg.common.model_type == "svdq":
        embedding_dim = cfg.svdq.embedding_dim
        svdq = TensorDictModule(
            SVDQ(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim, ),
            in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
            out_keys=["action_value"])
        return svdq

    randomq = TensorDictModule(
        Random(num_items=num_items,),
        in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
        out_keys=["action_value"])
    return randomq


def get_wandb_run(cfg, wandb):
    return wandb.init(
        project=cfg.common.wandb_project,
        config=dict(
            **OmegaConf.to_container(cfg.common, resolve=True, throw_on_missing=True),
            **OmegaConf.to_container(cfg[cfg.common.model_type], resolve=True, throw_on_missing=True)))


def train(cfg, wandb, logger):
    # Set random seed
    seed = cfg.common.seed
    torch.manual_seed(cfg.common.seed)
    random.seed(cfg.common.seed)
    np.random.seed(cfg.common.seed)

    # Preprocess data
    logger.info("Preprocess data")
    df, num_users, num_items, item_id_pad = get_dataset(cfg.common.dataset_name)
    users_pos_items = get_users_pos_items(df, num_users, item_id_pad)
    users_items_to_take_actions = get_users_items_to_take_actions(df, num_users, item_id_pad)
    interactions_graph = get_interactions_graph(df, num_items)
    #pos_interactions_graph = get_pos_interactions_graph(df, NUM_ITEMS)

    # Simple tests
    logger.info("Run simple tests")
    env = GCQNEnv(num_users, num_items, item_id_pad, df, users_pos_items, device="cpu")
    check_env_specs(env)
    env = GCQNEnv(num_users, num_items, item_id_pad, df, users_pos_items, device="cpu", env_for_valid=True)
    check_env_specs(env)
    logger.info(
        "Test QValueModule: " +
        test_QValueModule(num_users, num_items, item_id_pad, df, users_pos_items, users_items_to_take_actions))

    # Set environments
    logger.info(
        "Config common: \n"
        + pprint.pformat(OmegaConf.to_container(cfg.common, resolve=True, throw_on_missing=True), sort_dicts=False))
    device = torch.device(cfg.common.device)
    batch_size = cfg.common.batch_size
    test_size = cfg.common.test_size
    n_valid_episodes = cfg.common.n_valid_episodes
    train_users_ids, valid_users_ids = train_test_split(df.user_id.unique(), test_size=test_size, random_state=seed)
    train_env = GCQNEnv(num_users, num_items, item_id_pad, df[df.user_id.isin(train_users_ids)], users_pos_items,
                        batch_size=batch_size, device=device, seed=seed)
    valid_env = GCQNEnv(num_users, num_items, item_id_pad, df[df.user_id.isin(valid_users_ids)], users_pos_items,
                        batch_size=(test_size // n_valid_episodes), device=device, seed=seed, env_for_valid=True,)

    # Set model, optimizer, scheduler
    n_episodes = cfg.common.n_episodes
    e_greedy_eps_init = cfg.common.e_greedy_eps_init
    #exploration_eps = 0.2
    annealing_num_steps = cfg.common.annealing_num_steps
    updater_eps = cfg.common.soft_updater_eps
    value_network_update_interval = cfg.common.hard_updater_value_network_update_interval
    lr = cfg.common.optimizer_lr
    #patience = 10

    logger.info(
        "Config model: \n"
        + pprint.pformat(OmegaConf.to_container(cfg[cfg.common.model_type], resolve=True, throw_on_missing=True), sort_dicts=False))
    model = get_model(cfg, num_users, num_items, item_id_pad, df, interactions_graph, device)
    logger.info(f"Number trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # qval_with_exploration = GCQNQValueModule(df, users_items_to_take_actions, action_space=train_env.action_spec,
    #                                          exploration_eps=exploration_eps)
    qval = GCQNQValueModule(num_users=num_users, num_items=num_items, item_id_pad=item_id_pad,
                            interactions_df=df, users_items_to_take_actions=users_items_to_take_actions,
                            action_space=train_env.action_spec)
    stoch_policy = TensorDictSequential(model, qval).to(device)
    stoch_policy = EGreedyWrapper(
        stoch_policy, annealing_num_steps=annealing_num_steps, spec=train_env.action_spec, eps_init=e_greedy_eps_init,
    ).to(device)
    policy = TensorDictSequential(model, qval).to(device)
    # stoch_policy = TensorDictSequential(gcqn, qval).to(device)
    # policy = TensorDictSequential(gcqn, qval).to(device)

    criterion = DQNLoss(policy, action_space=train_env.action_spec, delay_value=True)
    if cfg.common.updater_type == "hard":
        updater = HardUpdate(criterion, value_network_update_interval=value_network_update_interval)
    else:
        updater = SoftUpdate(criterion, eps=updater_eps)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", verbose=True)

    # Init wandb
    run = get_wandb_run(cfg, wandb)

    # Train model
    logger.info("Run training")
    episode = 1
    for episode in tqdm.tqdm(range(episode, n_episodes + 1), desc="Training episodes"):
        train_time = time.time()
        train_loss_val, train_metric_val = train_loop(stoch_policy, train_env, optimizer, criterion, wandb=wandb,
                                                 episode=episode)
        train_time = time.time() - train_time
        stoch_policy.step()
        updater.step()
        if episode % 10 == 0:
            valid_loss_val, valid_metric_val = valid_loop(policy, valid_env, criterion, n_valid_episodes)
            scheduler.step(valid_metric_val)
            wandb.log({
                "Episode": episode,
                "train Time(s)": train_time,
                "train Loss": train_loss_val,
                "train AvgReturn": train_metric_val,
                "valid Loss": valid_loss_val,
                "valid AvgReturn": valid_metric_val,
            })
            logger.info(
                f"|Episode {episode}| Train Loss: {train_loss_val:.3f}; Train Metric: {train_metric_val:.3f}; "
                f"Valid Loss: {valid_loss_val:.3f}; Valid Metric: {valid_metric_val:.3f}; Train time: {train_time:.3f}s")
        else:
            wandb.log({
                "Episode": episode,
                "train Time(s)": train_time,
                "train Loss": train_loss_val,
                "train AvgReturn": train_metric_val,
            })
            logger.info(
                f"|Episode {episode}| Train Loss: {train_loss_val:.3f}; Train Metric: {train_metric_val:.3f}; Train time: {train_time:.3f}s")
    logger.info("Training completed")
    run.finish()
