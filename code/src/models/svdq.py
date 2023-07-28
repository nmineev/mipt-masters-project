import torch.nn as nn


class SVDQ(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.users_embeddings = nn.Embedding(num_users, embedding_dim)
        self.items_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, users_ids, users_items, users_rewards, step_num):
        users_embeddings = self.users_embeddings(users_ids.squeeze())
        return users_embeddings @ self.items_embeddings.weight.T
