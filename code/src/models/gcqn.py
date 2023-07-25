import torch
import torch.nn as nn
from graph_models_common import EmbeddingModule, MLPPredictor, DotPredictor


class GCQN(nn.Module):
    def __init__(self, num_items, num_users, interactions_graph, raw_embedding_dim, gnn_embedding_dim, rnn_embedding_dim,
                 reward_dim=16, num_heads=4, num_layers=1, num_neighbours=10, gnn_type="gat", rnn_type="gru",
                 predictor_type="dot", use_rewards_in_rnn=True, predictor_hidden_dim=100, device="cpu"):
        super().__init__()
        self.reward_dim = reward_dim
        self.items_embeddings = nn.Embedding(num_items, raw_embedding_dim)
        self.users_embeddings = nn.Embedding(num_users, raw_embedding_dim)

        in_node_feats, in_edge_feats, out_node_feats, out_edge_feats = raw_embedding_dim, reward_dim, gnn_embedding_dim, reward_dim
        self.embedding_module = EmbeddingModule(
            interactions_graph, in_node_feats, in_edge_feats, num_heads,
            out_node_feats, out_edge_feats, num_layers, num_neighbours, gnn_type, device=device)

        self.use_rewards_in_rnn = use_rewards_in_rnn
        rnn_input_dim = gnn_embedding_dim + self.reward_dim if self.use_rewards_in_rnn else gnn_embedding_dim
        self.rnn = nn.LSTM(rnn_input_dim, rnn_embedding_dim, batch_first=True) \
            if rnn_type == "lstm" else nn.GRU(rnn_input_dim, rnn_embedding_dim, batch_first=True)

        self.predictor = MLPPredictor(rnn_embedding_dim, gnn_embedding_dim, predictor_hidden_dim) \
            if predictor_type == "mlp" else DotPredictor()

    def forward(self, users_ids, users_items, users_rewards, step_num):
        step_num = step_num[0].item()
        users_ids = users_ids.squeeze()
        self.embedding_module.reset_interactions_graph()  # users_ids)
        #         self.embedding_module.add_interactions(users_ids,
        #                                                users_items[:, :step_num],
        #                                                users_rewards[:, :step_num])
        _, gnn_items_embeddings = self.embedding_module(torch.empty(0, dtype=users_ids.dtype, device=users_ids.device),
                                                        self.users_embeddings.weight,
                                                        self.items_embeddings.weight)
        users_items_embeddings = gnn_items_embeddings[users_items[:, :step_num]]
        if self.use_rewards_in_rnn:
            users_items_embeddings = torch.cat([users_items_embeddings,
                                                users_rewards[:, :step_num].unsqueeze(-1).repeat(1, 1,
                                                                                                 self.reward_dim).float()],
                                               dim=-1)
        _, users_embeddings = self.rnn(users_items_embeddings)
        if isinstance(self.rnn, nn.LSTM):
            users_embeddings = users_embeddings[0]
        users_embeddings = users_embeddings.squeeze()

        return self.predictor(users_embeddings, gnn_items_embeddings)
