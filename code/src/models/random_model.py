import torch
import torch.nn as nn


class Random(nn.Module):
    def __init__(self, num_items):
        self.num_items = num_items

    def forward(self, users_ids, users_items, users_rewards, step_num):
        return torch.rand((users_ids.size(0), self.num_items), device=users_ids.device)
