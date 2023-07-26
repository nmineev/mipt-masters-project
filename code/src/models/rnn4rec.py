import torch
import torch.nn as nn


class ItemsRNN(nn.Module):
    def __init__(self, num_items, embedding_dim=16, rnn_type="gru", reward_dim=16,
                 use_rewards=False):
        super().__init__()
        self.reward_dim = reward_dim
        self.use_rewards = use_rewards
        rnn_input_dim = embedding_dim + self.reward_dim if use_rewards else embedding_dim
        self.rnn = nn.LSTM(rnn_input_dim, embedding_dim, batch_first=True) \
            if rnn_type == "lstm" else nn.GRU(rnn_input_dim, embedding_dim, batch_first=True)
        self.items_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, users_ids, users_items, users_rewards, step_num):
        step_num = step_num[0].item()
        users_items_to_embed = users_items[:, :step_num].clone()
        users_items_embeddings = self.items_embeddings(users_items_to_embed)
        if self.use_rewards:
            users_items_embeddings = torch.cat([users_items_embeddings,
                                                users_rewards[:, :step_num].unsqueeze(-1).repeat(1, 1,
                                                                                                 self.reward_dim).float()],
                                               dim=-1)
        _, users_embeddings = self.rnn(users_items_embeddings)
        if isinstance(self.rnn, nn.LSTM):
            users_embeddings = users_embeddings[0]
        return users_embeddings.squeeze() @ self.items_embeddings.weight.T
