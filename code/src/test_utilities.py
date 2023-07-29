import copy
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from .environment import GCQNEnv
from .models import GCQNQValueModule, SVDQ


class PerfectRecommender(nn.Module):
    def __init__(self, num_users, num_items, users_pos_items):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(100.))
        self.num_users = num_users
        self.num_items = num_items
        self.users_pos_items = users_pos_items

    def forward(self, users_ids):
        batch_users_pos_items = self.users_pos_items[users_ids, :20]
        action_value = torch.zeros((users_ids.shape[0], self.num_items), dtype=torch.float)
        action_value[torch.arange(action_value.size(0)).unsqueeze(1), batch_users_pos_items] = self.param + 100.
        return action_value


# Perfomance tests
veq = torch.vmap(torch.eq)
def veq_test(recommended_items, users_items, veq=veq):
    return veq(recommended_items.unsqueeze(0).repeat(users_items.shape[1], 1, 1), users_items.T.unsqueeze(-1)).sum(0).bool()

visin = torch.vmap(torch.isin)
def visin_test(recommended_items, users_items, visin=visin):
    return visin(recommended_items, users_items)

def leq_test(recommended_items, users_items):
    return sum(recommended_items.eq(users_items_col.view(-1, 1)) for users_items_col in users_items.T).bool()

def test_correctness(recommended_items, users_items):
    assert ((veq_test(recommended_items, users_items) == visin_test(recommended_items, users_items))
            == leq_test(recommended_items, users_items)).all().item()

# recommended_items = torch.randint(0, 5, (1000, 1000))
# users_items = torch.randint(0, 5, (1000, 100))

# print(recommended_items.shape, users_items.shape)

# test_correctness(recommended_items, users_items)

# %timeit veq_test(recommended_items, users_items)
# %timeit visin_test(recommended_items, users_items)
# %timeit leq_test(recommended_items, users_items)


@torch.no_grad()
def test_QValueModule(num_users, num_items, item_id_pad, df, users_pos_items, users_items_to_take_actions):
    env = GCQNEnv(num_users, num_items, item_id_pad, df, users_pos_items, batch_size=100)
    svd = TensorDictModule(SVDQ(num_users, num_items, 10),
                           in_keys=["users_ids", "users_items", "users_rewards", "step_num"],
                           out_keys=["action_value"])
    qval = GCQNQValueModule(num_users, num_items, item_id_pad, df, users_items_to_take_actions, action_space=env.action_spec)
    td = env.reset()
    users_ids = td["users_ids"]
    first_items = td["users_items"][:, 0]
    pos_items_ids = torch.zeros_like(users_ids)
    neg_items_ids = torch.zeros_like(users_ids)
    for user_num, user_id in enumerate(users_ids):
        user_id = user_id.item()
        first_item = first_items[user_num].item()
        pos_items_ids[user_num] = \
        df[(df.user_id == user_id) & (df.item_id != first_item) & (df.reward == 1)].item_id.iloc[0]
        if len(df[(df.user_id == user_id) & (df.item_id != first_item) & (df.reward == 0)].item_id):
            neg_items_ids[user_num] = \
            df[(df.user_id == user_id) & (df.item_id != first_item) & (df.reward == 0)].item_id.iloc[0]
    td = svd(td)

    td["action_value"] *= 0.
    for user_num, user_id in enumerate(users_ids):
        td["action_value"][user_num, pos_items_ids[user_num]] = 100.
    td_pos = qval(copy.deepcopy(td))
    td_pos = env.step(td_pos)
    assert (td_pos["action"] == pos_items_ids).all()
    assert td_pos["next", "reward"].all()

    td["action_value"] *= 0.
    for user_num, user_id in enumerate(users_ids):
        td["action_value"][user_num, neg_items_ids[user_num]] = 100.
    td_pos = qval(copy.deepcopy(td))
    td_pos = env.step(td_pos)
    assert (td_pos["action"] == neg_items_ids)[neg_items_ids != 0].all()
    assert not td_pos["next", "reward"].all()

    td["action_value"] *= 0.
    for user_num, user_id in enumerate(users_ids):
        td["action_value"][user_num, first_items[user_num]] = 101.
        td["action_value"][user_num, pos_items_ids[user_num]] = 100.
    td_pos = qval(copy.deepcopy(td))
    td_pos = env.step(td_pos)
    assert (td_pos["action"] == pos_items_ids).all()
    assert td_pos["next", "reward"].all()

    td["action_value"] *= 0.
    for user_num, user_id in enumerate(users_ids):
        td["action_value"][user_num, first_items[user_num]] = 101.
        td["action_value"][user_num, neg_items_ids[user_num]] = 100.
    td_pos = qval(copy.deepcopy(td))
    td_pos = env.step(td_pos)
    assert (td_pos["action"] == neg_items_ids)[neg_items_ids != 0].all()
    assert not td_pos["next", "reward"].all()

    qval_exploration = GCQNQValueModule(num_users, num_items, item_id_pad, df, users_items_to_take_actions, action_space=env.action_spec,
                                        exploration_eps=5.,
                                        num_to_take_in_exploration=20)
    td = env.reset()
    td = svd(td)
    td = qval_exploration(td)
    assert torch.isin(td["action"][0], users_items_to_take_actions[td["users_ids"][0], :100]).item() \
           and not torch.isin(td["action"][0], users_items_to_take_actions[td["users_ids"][0], 100:400]).item()
    assert torch.isin(td["action"][50], users_items_to_take_actions[td["users_ids"][50], :100]).item() \
           and not torch.isin(td["action"][50], users_items_to_take_actions[td["users_ids"][50], 100:400]).item()
    assert torch.isin(td["action"][75], users_items_to_take_actions[td["users_ids"][75], :100]).item() \
           and not torch.isin(td["action"][75], users_items_to_take_actions[td["users_ids"][75], 100:400]).item()

    return "success"



# Offline Training for debug
# user_item_reward_matrix = torch.zeros((NUM_USERS, NUM_ITEMS))
# user_pos_reward_inds = torch.from_numpy(df[df.reward == 1].user_id.values)
# item_pos_reward_inds = torch.from_numpy(df[df.reward == 1].item_id.values)
# user_item_reward_matrix[user_pos_reward_inds, item_pos_reward_inds] = 1
# user_item_reward_matrix = user_item_reward_matrix.to(device)
#
# pos_weight = (user_item_reward_matrix.numel() - df[df.reward == 1].shape[0]) / df[df.reward == 1].shape[0]


# def BCEOfflineCriterion(input_tensordict):
#     loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
#     users_ids = input_tensordict["users_ids"]
#     logits = input_tensordict["action_value"]
#     target = user_item_reward_matrix[users_ids]
#
#     return {"loss": loss(logits, target)}
