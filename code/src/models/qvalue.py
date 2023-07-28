import numpy as np
import torch
from torchrl.modules import QValueModule

visin = torch.vmap(torch.isin)


class GCQNQValueModule(QValueModule):
    def __init__(self, num_users, num_items, item_id_pad, interactions_df, users_items_to_take_actions,
                 num_top_to_take_action=100, exploration_eps=0, num_to_take_in_exploration=50,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions_df = interactions_df
        self.users_items_to_take_actions = users_items_to_take_actions
        self.num_top_to_take_action = num_top_to_take_action
        self.exploration_eps = exploration_eps
        self.num_to_take_in_exploration = num_to_take_in_exploration
        self.num_items = num_items
        self.num_users = num_users
        self.item_id_pad = item_id_pad

    def forward(self, input_tensordict):
        device = input_tensordict.device

        action_values = input_tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {input_tensordict}."
            )

        sampled_items_ids_to_take_actions = self.users_items_to_take_actions[
            input_tensordict["users_ids"].cpu()].clone()
        pad_mask = (sampled_items_ids_to_take_actions == self.item_id_pad)
        sampled_items_ids_to_take_actions[pad_mask] = torch.randint(0, self.num_items, (pad_mask.sum(),))
        sampled_items_ids_to_take_actions = sampled_items_ids_to_take_actions.to(device)
        sampled_action_values = action_values.gather(1, sampled_items_ids_to_take_actions)
        top_sampled_items_ids_to_take_actions_inds = sampled_action_values.topk(self.num_top_to_take_action, 1).indices
        top_sampled_items_ids_to_take_actions = sampled_items_ids_to_take_actions.gather(1,
                                                                                         top_sampled_items_ids_to_take_actions_inds)

        action = top_sampled_items_ids_to_take_actions.gather(1, visin(top_sampled_items_ids_to_take_actions,
                                                                       input_tensordict["users_items"]).int().argmin(
            1).view(-1, 1))
        if np.random.uniform() < self.exploration_eps:
            batch_size = input_tensordict["users_ids"].size(0)
            rand_inds = torch.randint(0, self.num_to_take_in_exploration,
                                      size=(batch_size,), device=device)
            action = sampled_items_ids_to_take_actions[torch.arange(batch_size, device=device), rand_inds]

        action = action.squeeze()
        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        input_tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return input_tensordict

#     def forward_old(self, input_tensordict):
#         device = input_tensordict.device
#         self.num_items_to_take_action = self.users_items_to_take_actions.shape[1]
#         action_values = input_tensordict.get(self.action_value_key, None)
#         if action_values is None:
#             raise KeyError(
#                 f"Action value key {self.action_value_key} not found in {input_tensordict}."
#             )

#         sampled_items_ids_to_take_actions = torch.empty((input_tensordict.batch_size[0], self.num_items_to_take_action),
#                                                          dtype=torch.long)
#         #print("sampled_items_ids_to_take_actions", sampled_items_ids_to_take_actions[4])
#         items_ids = torch.arange(NUM_ITEMS, dtype=torch.long)
#         for user_num, user_id in enumerate(input_tensordict["users_ids"]):
#             user_id = user_id.item()
#             user_interacted_items_ids = torch.from_numpy(self.interactions_df[self.interactions_df.user_id == user_id].item_id.values)
#             if len(user_interacted_items_ids) >= self.num_items_to_take_action:
#                 user_interacted_items_ids_inds = torch.randperm(len(user_interacted_items_ids))[:self.num_items_to_take_action]
#                 user_sampled_items_ids_to_take_action = user_interacted_items_ids[user_interacted_items_ids_inds]
#             else:
#                 user_not_interacted_items_ids_mask = torch.isin(items_ids, user_interacted_items_ids, assume_unique=True, invert=True)
#                 user_not_interacted_items_ids = items_ids[user_not_interacted_items_ids_mask]
#                 user_not_interacted_items_ids_inds = \
#                 torch.randperm(len(user_not_interacted_items_ids))[:self.num_items_to_take_action - len(user_interacted_items_ids)]
#                 user_not_interacted_items_ids = user_not_interacted_items_ids[user_not_interacted_items_ids_inds]
#                 user_sampled_items_ids_to_take_action = torch.cat([user_interacted_items_ids, user_not_interacted_items_ids], 0)
#             sampled_items_ids_to_take_actions[user_num] = user_sampled_items_ids_to_take_action
#         sampled_items_ids_to_take_actions = sampled_items_ids_to_take_actions.to(device)
#         #print("sampled_items_ids_to_take_actions", sampled_items_ids_to_take_actions[4])
#         sampled_action_values = action_values.gather(1, sampled_items_ids_to_take_actions)
#         #print("sampled_action_values", sampled_action_values[4])
#         top_sampled_items_ids_to_take_actions_inds = sampled_action_values.topk(self.num_top_to_take_action, 1).indices
#         #print("top_sampled_items_ids_to_take_actions_inds", top_sampled_items_ids_to_take_actions_inds[4])
#         top_sampled_items_ids_to_take_actions = sampled_items_ids_to_take_actions.gather(1, top_sampled_items_ids_to_take_actions_inds)
#         #print("top_sampled_items_ids_to_take_actions", top_sampled_items_ids_to_take_actions[4])

#         action = torch.zeros_like(input_tensordict["users_ids"])
#         for user_num, user_id in enumerate(input_tensordict["users_ids"]):
#             action[user_num] = top_sampled_items_ids_to_take_actions[user_num, 0]
#             new_recommended_items_mask = torch.isin(top_sampled_items_ids_to_take_actions[user_num],
#                                                     input_tensordict["users_items"][user_num],
#                                                     assume_unique=True, invert=True)
#             new_recommended_items = top_sampled_items_ids_to_take_actions[user_num, new_recommended_items_mask]
#             if len(new_recommended_items):
#                 action[user_num] = new_recommended_items[0]
#         #print("action", action[4])
#         action_value_func = self.action_value_func_mapping.get(
#             self.action_space, self._default_action_value
#         )
#         chosen_action_value = action_value_func(action_values, action)
#         #print("chosen_action_value", chosen_action_value[4])
#         input_tensordict.update(
#             dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
#         )
#         return input_tensordict
