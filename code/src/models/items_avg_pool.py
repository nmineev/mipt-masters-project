import torch.nn as nn


class ItemsAvgPool(nn.Module):
    def __init__(self, num_items, item_id_pad, embedding_dim):
        super().__init__()
        self.item_id_pad = item_id_pad
        self.items_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=self.item_id_pad)

    def forward(self, users_items, users_rewards, step_num):
        step_num = step_num[0].item()
        users_items_to_embed = users_items[:, :step_num].clone()
        users_items_to_embed[~users_rewards[:, :step_num].bool()] = self.item_id_pad
        users_embeddings = self.items_embeddings(users_items_to_embed).mean(1)
        return users_embeddings @ self.items_embeddings.weight.T
