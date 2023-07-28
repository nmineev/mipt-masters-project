import torch
import torch.nn as nn
from graph_models_common import MemoryModule, EmbeddingModule, MLPPredictor, DotPredictor


class TGQN(nn.Module):
    def __init__(self, num_users, num_items, interactions_graph, memory_dim, embedding_dim, reward_dim, num_heads=4,
                 num_layers=1, num_neighbours=10, gnn_type="edge_gat", rnn_type="gru", memory_alpha=0.5,
                 use_users_raw_embeddings=False, items_memory_batch_agg="mean", items_embedding_module_input="sum",
                 users_embedding_module_input="sum", predictor_type="dot", predictor_hidden_dim=100,
                 users_predictor_input="embedding", items_predictor_input="embedding", use_items_memory_as_hidden=False,
                 pos_interactions_only=False, dropout=0.2, device="cpu"):
        super().__init__()
        if not use_users_raw_embeddings:
            users_embedding_module_input = "memory"

        if (items_memory_batch_agg not in ["sum", "mean", "last"]) \
                or (users_embedding_module_input not in ["sum", "cat", "memory", "raw_embedding"]) \
                or (items_embedding_module_input not in ["sum", "cat", "memory", "raw_embedding"]) \
                or (users_predictor_input not in ["sum", "cat", "memory", "raw_embedding", "embedding"]) \
                or (items_predictor_input not in ["sum", "cat", "memory", "raw_embedding", "embedding"]) \
                or (items_embedding_module_input == "cat" and users_embedding_module_input != "cat") \
                or (items_embedding_module_input != "cat" and users_embedding_module_input == "cat") \
                or (items_predictor_input == "sum" and memory_dim != embedding_dim) \
                or (users_predictor_input == "sum" and memory_dim != embedding_dim):
            raise RuntimeError("Wrong input")

        self.reward_dim = reward_dim
        self.use_users_raw_embeddings = use_users_raw_embeddings
        self.items_memory_batch_agg = items_memory_batch_agg
        self.items_embedding_module_input = items_embedding_module_input
        self.users_embedding_module_input = users_embedding_module_input
        self.users_predictor_input = users_predictor_input
        self.items_predictor_input = items_predictor_input
        self.use_items_memory_as_hidden = use_items_memory_as_hidden

        self.items_raw_embeddings = nn.Embedding(num_items, memory_dim)
        if use_users_raw_embeddings:
            self.users_raw_embeddings = nn.Embedding(num_users, memory_dim)

        self.memory_module = MemoryModule(memory_dim, memory_alpha)

        rnn_input_dim = 2 * memory_dim + self.reward_dim if use_users_raw_embeddings else memory_dim + self.reward_dim
        self.rnn = nn.LSTM(rnn_input_dim, memory_dim, batch_first=True) \
            if rnn_type == "lstm" else nn.GRU(rnn_input_dim, memory_dim, batch_first=True)

        in_node_feats = memory_dim * 2 if items_embedding_module_input == "cat" else memory_dim
        in_edge_feats, out_node_feats, out_edge_feats = reward_dim, embedding_dim, reward_dim
        self.embedding_module = EmbeddingModule(
            interactions_graph, in_node_feats, in_edge_feats, num_heads,
            out_node_feats, out_edge_feats, num_layers, num_neighbours, gnn_type, pos_interactions_only, device=device)

        predictor_input_dims = {"sum": memory_dim, "cat": memory_dim * 2 + embedding_dim, "memory": memory_dim,
                                "raw_embedding": memory_dim, "embedding": embedding_dim}
        users_predictor_input_dim = predictor_input_dims[users_predictor_input] \
                                    - ((users_predictor_input == "cat") and (not use_users_raw_embeddings)) * memory_dim
        items_predictor_input_dim = predictor_input_dims[items_predictor_input]
        self.predictor = MLPPredictor(users_predictor_input_dim, items_predictor_input_dim, predictor_hidden_dim) \
            if predictor_type == "mlp" else DotPredictor()

        self.dropout = nn.Dropout(dropout)

    def forward(self, users_ids, users_items, users_rewards, step_num):
        step_num = step_num[0].item()
        users_ids = users_ids.squeeze()
        self.memory_module.restore_memory_from_backup()
        self.embedding_module.reset_interactions_graph(users_ids)
        self.embedding_module.add_interactions(users_ids,
                                               users_items[:, :step_num],
                                               users_rewards[:, :step_num])
        users_items_to_embed = users_items[:, :step_num].clone()
        users_items_raw_embeddings = self.dropout(self.items_raw_embeddings(users_items_to_embed))
        if self.use_users_raw_embeddings:
            users_raw_embeddings = self.dropout(self.users_raw_embeddings(users_ids.view(-1, 1).repeat(1, step_num)))
            users_items_embeddings = torch.cat([users_items_raw_embeddings,
                                                users_raw_embeddings,
                                                users_rewards[:, :step_num].unsqueeze(-1).repeat(1, 1,
                                                                                                 self.reward_dim).float()],
                                               dim=-1)
        else:
            users_items_embeddings = torch.cat([users_items_raw_embeddings,
                                                users_rewards[:, :step_num].unsqueeze(-1).repeat(1, 1,
                                                                                                 self.reward_dim).float()],
                                               dim=-1)
        sequence_items_memory, users_memory = self.rnn(users_items_embeddings)
        if isinstance(self.rnn, nn.LSTM):
            users_memory = users_memory[0]
        users_memory = users_memory.squeeze()
        self.memory_module.set_memory(users_ids, users_memory)

        if self.use_items_memory_as_hidden:
            last_items = users_items_to_embed[:, -1].squeeze()
            last_items_memory = self.memory_module.items_memory[last_items].unsqueeze(0)
            rnn_input = users_items_embeddings[:, -1, :].unsqueeze(1)
            if isinstance(self.rnn, nn.LSTM):
                _, items_memory = self.rnn(rnn_input, (last_items_memory, torch.zeros_like(last_items_memory)))
                items_memory = items_memory[0]
            else:
                _, items_memory = self.rnn(rnn_input, last_items_memory)
            items_memory = items_memory.squeeze()
            self.memory_module.items_memory[last_items] = items_memory
            with torch.no_grad():
                self.memory_module.items_memory_backup[last_items] = items_memory

        else:
            items_ids, items_inds, items_counts = users_items_to_embed.unique(return_inverse=True, return_counts=True)
            items_memory = torch.zeros((items_ids.size(0), sequence_items_memory.size(2)),
                                       dtype=sequence_items_memory.dtype,
                                       device=sequence_items_memory.device)
            if self.items_memory_batch_agg in ["sum", "mean"]:
                items_memory.index_add_(0, items_inds.flatten(), sequence_items_memory.flatten(0, 1))
                if self.items_memory_batch_agg == "mean":
                    items_memory /= items_counts.view(-1, 1)
            else:
                items_memory[items_inds] += sequence_items_memory
            self.memory_module.update_memory(items_ids, items_memory)

        if step_num == 19 and self.training:
            if self.use_items_memory_as_hidden:
                items_ids = None
            self.memory_module.update_backup(users_ids, users_memory, items_ids, items_memory)

        if self.users_embedding_module_input == "sum":
            users_embedding_module_input = self.memory_module.users_memory + self.users_raw_embeddings.weight
        elif self.users_embedding_module_input == "cat":
            users_embedding_module_input = torch.cat(
                [self.memory_module.users_memory, self.users_raw_embeddings.weight], dim=1)
        elif self.users_embedding_module_input == "memory":
            users_embedding_module_input = self.memory_module.users_memory
        elif self.users_embedding_module_input == "raw_embedding":
            users_embedding_module_input = self.users_raw_embeddings.weight

        if self.items_embedding_module_input == "sum":
            items_embedding_module_input = self.memory_module.items_memory + self.items_raw_embeddings.weight
        elif self.items_embedding_module_input == "cat":
            items_embedding_module_input = torch.cat(
                [self.memory_module.items_memory, self.items_raw_embeddings.weight], dim=1)
        elif self.items_embedding_module_input == "memory":
            items_embedding_module_input = self.memory_module.items_memory
        elif self.items_embedding_module_input == "raw_embedding":
            items_embedding_module_input = self.items_raw_embeddings.weight

        users_embeddings, items_embeddings = self.embedding_module(users_ids,
                                                                   users_embedding_module_input,
                                                                   items_embedding_module_input)

        if self.users_predictor_input == "sum":
            users_predictor_input = self.memory_module.users_memory[users_ids] + users_embeddings
            if self.use_users_raw_embeddings:
                users_predictor_input += self.users_raw_embeddings(users_ids)
        elif self.users_predictor_input == "cat":
            users_predictor_input = [self.memory_module.users_memory[users_ids], users_embeddings]
            if self.use_users_raw_embeddings:
                users_predictor_input.append(self.users_raw_embeddings(users_ids))
            users_predictor_input = torch.cat(users_predictor_input, dim=1)
        elif self.users_predictor_input == "memory":
            users_predictor_input = self.memory_module.users_memory[users_ids]
        elif self.users_predictor_input == "raw_embedding":
            users_predictor_input = self.users_raw_embeddings(users_ids)
        elif self.users_predictor_input == "embedding":
            users_predictor_input = users_embeddings

        if self.items_predictor_input == "sum":
            items_predictor_input = self.memory_module.items_memory + items_embeddings + self.items_raw_embeddings.weight
        elif self.items_predictor_input == "cat":
            items_predictor_input = [self.items_raw_embeddings.weight, items_embeddings,
                                     self.memory_module.items_memory]
            items_predictor_input = torch.cat(items_predictor_input, dim=1)
        elif self.items_predictor_input == "memory":
            items_predictor_input = self.memory_module.items_memory
        elif self.items_predictor_input == "raw_embedding":
            items_predictor_input = self.items_raw_embeddings.weight
        elif self.items_predictor_input == "embedding":
            items_predictor_input = items_embeddings

        return self.predictor(self.dropout(users_predictor_input), self.dropout(items_predictor_input))