import torch
from torchrl.envs import EnvBase
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedDiscreteTensorSpec


class GCQNEnv(EnvBase):
    def __init__(self, interactions_df, users_pos_items, NUM_ITEMS, NUM_USERS, ITEM_ID_PAD,
                 batch_size=32, episode_num_steps=20, seed=1488, device="cpu", env_for_valid=False):
        super().__init__(device=device, batch_size=[batch_size])
        self.interactions_df = interactions_df
        self.users_ids = interactions_df.user_id.unique()
        self.episode_num_steps = episode_num_steps
        self.users_pos_items = users_pos_items
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.env_for_valid = env_for_valid
        self.batch_num = 0
        self.NUM_ITEMS = NUM_ITEMS
        self.NUM_USERS = NUM_USERS
        self.ITEM_ID_PAD = ITEM_ID_PAD

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, input_tensordict):
        if self.env_for_valid:
            if (self.batch_num + 1) * self.batch_size[0] > len(self.users_ids):
                self.batch_num = 0
            batch_users_ids = self.users_ids[
                              self.batch_num * self.batch_size[0]:(self.batch_num + 1) * self.batch_size[0]]
            self.batch_num += 1
        else:
            batch_users_inds = torch.randperm(len(self.users_ids), generator=self.rng)[:self.batch_size[0]]
            batch_users_ids = self.users_ids[
                batch_users_inds]  # np.random.choice(self.users_ids, size=self.batch_size[0], replace=False)
        batch_users_items = torch.full((self.batch_size[0], self.episode_num_steps), self.ITEM_ID_PAD, dtype=torch.int64)
        batch_users_timestamps = torch.zeros(self.batch_size[0], self.episode_num_steps, dtype=torch.int64)
        batch_users_rewards = torch.zeros(self.batch_size[0], self.episode_num_steps, dtype=torch.int64)
        for user_num, user_id in enumerate(batch_users_ids):
            user_interactions = self.interactions_df[self.interactions_df.user_id == user_id]
            episode_start_interaction_ind = torch.randint(0, len(user_interactions) - self.episode_num_steps + 1,
                                                          size=(1,), generator=self.rng).item()
            # np.random.randint(len(user_interactions) - self.episode_num_steps + 1)
            batch_users_items[user_num, 0] = user_interactions.iloc[episode_start_interaction_ind].item_id
            batch_users_rewards[user_num, 0] = user_interactions.iloc[episode_start_interaction_ind].reward
            batch_users_timestamps[user_num] = torch.from_numpy(
                user_interactions.iloc[episode_start_interaction_ind
                                       :episode_start_interaction_ind + self.episode_num_steps].timestamp.values)
        output_tensordict = TensorDict({
            "users_ids": batch_users_ids,
            "users_items": batch_users_items,
            "users_rewards": batch_users_rewards,
            "users_timestamps": batch_users_timestamps,
            "step_num": torch.ones(self.batch_size, dtype=torch.int64),
        }, batch_size=self.batch_size, device=self.device)
        return output_tensordict

    def _step(self, input_tensordict):
        users_ids = input_tensordict["users_ids"]
        users_items = input_tensordict["users_items"]
        users_rewards = input_tensordict["users_rewards"]
        users_timestamps = input_tensordict["users_timestamps"]
        step_num = input_tensordict["step_num"][0].item()
        recommended_items = input_tensordict["action"].view(-1, 1)
        reward = self.users_pos_items[users_ids.cpu()].to(self.device).eq(recommended_items).any(1)
        is_recommendation_new = ~users_items.eq(recommended_items).any(1)
        reward *= is_recommendation_new
        reward = reward.long()
        users_items[:, step_num] = recommended_items.squeeze()
        users_rewards[:, step_num] = reward
        done = torch.zeros_like(users_ids, dtype=torch.bool) \
            if step_num + 1 < self.episode_num_steps else torch.ones_like(users_ids, dtype=torch.bool)

        output_tensordict = TensorDict({
            "next": {
                "users_ids": users_ids,
                "users_items": users_items,
                "users_rewards": users_rewards,
                "users_timestamps": users_timestamps,
                "step_num": torch.full(self.batch_size, step_num + 1, dtype=torch.int64, device=self.device),
                "reward": users_rewards[:, step_num].view(-1, 1),
                "done": done.view(-1, 1),
            }
        }, batch_size=self.batch_size, device=self.device)
        return output_tensordict

    #     def _step_old(self, input_tensordict):
    #         users_ids = input_tensordict["users_ids"]
    #         users_items = input_tensordict["users_items"]
    #         users_rewards = input_tensordict["users_rewards"]
    #         users_timestamps = input_tensordict["users_timestamps"]
    #         step_num = input_tensordict["step_num"][0].item()
    #         recommended_items = input_tensordict["action"]
    #         #reward = self.users_pos_items[users_ids].eq(recommended_items.view(-1, 1)).any(1)
    #         for user_num, user_id in enumerate(users_ids):
    #             user_id = user_id.item()
    # #             new_recommended_items_mask = torch.isin(recommended_items[user_num], users_items[user_num],
    # #                                                     assume_unique=True, invert=True)
    # #             new_recommended_items = recommended_items[user_num, new_recommended_items_mask]
    #             users_rewards[user_num, step_num] = 0
    #             if recommended_items[user_num] not in users_items[user_num]:
    #                 recommended_item_id = recommended_items[user_num].item()
    #                 user_recommended_item_interaction = self.interactions_df[(self.interactions_df.user_id == user_id)
    #                                                                          & (self.interactions_df.item_id == recommended_item_id)]
    #                 if not user_recommended_item_interaction.empty:
    #                     users_rewards[user_num, step_num] = user_recommended_item_interaction.reward.item()
    #             users_items[user_num, step_num] = recommended_items[user_num]
    #         done = torch.zeros_like(users_ids, dtype=torch.bool) \
    #         if step_num + 1 < self.episode_num_steps else torch.ones_like(users_ids, dtype=torch.bool)

    #         output_tensordict = TensorDict({
    #             "next": {
    #                 "users_ids": users_ids,
    #                 "users_items": users_items,
    #                 "users_rewards": users_rewards,
    #                 "users_timestamps": users_timestamps,
    #                 "step_num": torch.full(self.batch_size, step_num + 1, dtype=torch.int64, device=self.device),
    #                 "reward": users_rewards[:, step_num].view(-1, 1),
    #                 "done": done.view(-1, 1),
    #             }
    #         }, batch_size=self.batch_size, device=self.device)
    #         return output_tensordict

    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            users_ids=DiscreteTensorSpec(
                n=self.NUM_USERS,
                shape=self.batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            users_items=DiscreteTensorSpec(
                n=self.NUM_ITEMS + 1,
                shape=(self.batch_size[0], self.episode_num_steps),
                dtype=torch.int64,
                device=self.device,
            ),
            users_rewards=DiscreteTensorSpec(
                n=2,
                shape=(self.batch_size[0], self.episode_num_steps),
                dtype=torch.int64,
                device=self.device,
            ),
            users_timestamps=UnboundedDiscreteTensorSpec(
                shape=(self.batch_size[0], self.episode_num_steps),
                dtype=torch.int64,
                device=self.device,
            ),
            step_num=UnboundedDiscreteTensorSpec(
                shape=self.batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = DiscreteTensorSpec(
            n=self.NUM_ITEMS,
            shape=self.batch_size,
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = DiscreteTensorSpec(
            n=2,
            shape=(self.batch_size[0], 1),
            dtype=torch.int64,
            device=self.device,
        )
