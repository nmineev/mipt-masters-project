import torch.nn as nn


class SVD(nn.Module):
    def __init__(self, embedding_dim, num_users, num_items):
        super().__init__()
        self.users_embeddings = nn.Embedding(num_users, embedding_dim)
        self.items_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, users_inds):
        users_embeddings = self.users_embeddings(users_inds.squeeze())
        return users_embeddings @ self.items_embeddings.weight.T
