import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn


class EGAT(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, num_heads, out_node_feats=None, out_edge_feats=None, num_layers=1):
        super().__init__()
        out_node_feats = in_node_feats if out_node_feats is None else out_node_feats
        out_edge_feats = in_edge_feats if out_edge_feats is None else out_edge_feats
        self.in_edge_feats = in_edge_feats
        self.convs = nn.ModuleList(
            [dglnn.EGATConv(in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads)])
        self.convs.extend([dglnn.EGATConv(out_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads)
                           for _ in range(num_layers - 1)])
        self.relu = nn.ReLU()

    def forward(self, blocks, nfeats):
        for i in range(len(self.convs)):
            efeats = blocks[i].edata["reward"].unsqueeze(-1).repeat(1, self.in_edge_feats).float()
            nfeats, _ = self.convs[i](blocks[i], (nfeats, blocks[i].dstdata["feats"]), efeats)
            nfeats = self.relu(nfeats.mean(-2))
        return nfeats


class EdgeGAT(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, num_heads, out_node_feats=None, num_layers=1):
        super().__init__()
        out_node_feats = in_node_feats if out_node_feats is None else out_node_feats
        self.in_edge_feats = in_edge_feats
        self.convs = nn.ModuleList([dglnn.EdgeGATConv(in_node_feats, in_edge_feats, out_node_feats, num_heads)])
        self.convs.extend([dglnn.EdgeGATConv(out_node_feats, in_edge_feats, out_node_feats, num_heads)
                           for _ in range(num_layers - 1)])
        self.relu = nn.ReLU()

    def forward(self, blocks, nfeats):
        for i in range(len(self.convs)):
            efeats = blocks[i].edata["reward"].unsqueeze(-1).repeat(1, self.in_edge_feats).float()
            nfeats = self.convs[i](blocks[i], nfeats, efeats)
            nfeats = self.relu(nfeats.mean(-2))
        return nfeats


class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_layers=1):
        super().__init__()
        self.convs = nn.ModuleList([dglnn.SAGEConv(in_feat, out_feat, "mean")])
        self.convs.extend([dglnn.SAGEConv(out_feat, out_feat, "mean") for _ in range(num_layers - 1)])
        self.relu = nn.ReLU()

    def forward(self, blocks, feats):
        for i in range(len(self.convs)):
            feats = self.relu(self.convs[i](blocks[i], feats))
        return feats


class GAT(nn.Module):
    def __init__(self, in_feat, out_feat, num_heads, num_layers=1):
        super().__init__()
        self.convs = nn.ModuleList([dglnn.GATConv(in_feat, out_feat, num_heads)])
        self.convs.extend([dglnn.GATConv(out_feat, out_feat, num_heads) for _ in range(num_layers - 1)])
        self.relu = nn.ReLU()

    def forward(self, blocks, feats):
        for i in range(len(self.convs)):
            feats = self.relu(self.convs[i](blocks[i], feats).mean(-2))
        return feats


class EmbeddingModule(nn.Module):
    def __init__(self, num_items, interactions_graph, in_node_feats, in_edge_feats=16, num_heads=4,
                 out_node_feats=None, out_edge_feats=None, num_layers=1, num_neighbours=10,
                 gnn_type="edge_gat", pos_interactions_only=False, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.out_node_feats = in_node_feats if out_node_feats is None else out_node_feats
        self.interactions_graph_backup = interactions_graph
        self.interactions_graph = None
        self.pos_interactions_only = pos_interactions_only
        self.gnn = None
        if gnn_type == "gat":
            self.gnn = GAT(in_node_feats, out_node_feats, num_heads, num_layers, )
        elif gnn_type == "gcn":
            self.gnn = GCN(in_node_feats, out_node_feats, num_layers)
        elif gnn_type == "egat":
            self.gnn = EGAT(in_node_feats, in_edge_feats, num_heads, out_node_feats, out_edge_feats, num_layers)
        else:
            self.gnn = EdgeGAT(in_node_feats, in_edge_feats, num_heads, out_node_feats, num_layers, )
        self.neighbour_sampler = dgl.dataloading.NeighborSampler(
            [num_neighbours for _ in range(num_layers)], replace=True)
        self.num_items = num_items

    def reset_interactions_graph(self, users_ids=None):
        device = self.device
        self.interactions_graph = self.interactions_graph_backup.clone().to(device)
        if users_ids is not None:
            nids_to_remove_edges = (users_ids + self.num_items).to(device)
            in_eids = self.interactions_graph.in_edges(nids_to_remove_edges, form="eid")
            out_eids = self.interactions_graph.out_edges(nids_to_remove_edges, form="eid")
            self.interactions_graph.remove_edges(torch.cat([in_eids, out_eids]))
            self.interactions_graph = self.interactions_graph.remove_self_loop()
            self.interactions_graph = self.interactions_graph.add_self_loop(fill_data=0)

    def add_interactions(self, users_ids, items_ids, rewards):
        device = self.device
        users_nids = (users_ids + self.num_items).repeat_interleave(items_ids.size(1)).to(device)
        items_nids = items_ids.flatten().to(device)
        rewards = rewards.flatten().to(device)
        if self.pos_interactions_only:
            users_nids = users_nids[rewards.bool()]
            items_nids = items_nids[rewards.bool()]
            rewards = rewards[rewards.bool()]
        self.interactions_graph.add_edges(
            torch.cat([users_nids, items_nids]),
            torch.cat([items_nids, users_nids]),
            data={"reward": torch.cat([rewards, rewards.clone()])})

    def forward(self, users_ids, users_features, items_features):
        device = self.device
        node_features = torch.cat([items_features, users_features], dim=0)
        self.interactions_graph.ndata["feats"] = node_features
        nodes_to_embed = torch.cat([torch.arange(self.num_items, device=device), users_ids + self.num_items])
        out = torch.zeros((len(nodes_to_embed), self.out_node_feats), device=device)
        batch_size = 1024
        dataloader = dgl.dataloading.DataLoader(
            self.interactions_graph, nodes_to_embed, self.neighbour_sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        for batch_num, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            nfeats = blocks[0].srcdata["feats"]
            out_embs = self.gnn(blocks, nfeats)
            out[batch_num * batch_size:batch_num * batch_size + batch_size] = out_embs
        return out[self.num_items:], out[:self.num_items]


class DotPredictor(nn.Module):
    def forward(self, users_embeddings, items_embeddings):
        return users_embeddings @ items_embeddings.T


class MLPPredictor(nn.Module):
    def __init__(self, user_embedding_dim, item_embedding_dim, hidden_dim=100, dropout=0.2, users_chunk_size=100):
        super().__init__()
        self.users_chunk_size = users_chunk_size
        self.user_fc_first = nn.Linear(user_embedding_dim, hidden_dim)
        self.item_fc_first = nn.Linear(item_embedding_dim, hidden_dim)
        self.fc_second = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, users_embeddings, items_embeddings):
        users_out = self.user_fc_first(users_embeddings)
        items_out = self.item_fc_first(items_embeddings)
        out = users_out.repeat_interleave(items_embeddings.size(0), dim=0) + items_out.repeat(users_embeddings.size(0),
                                                                                              1)
        out = self.fc_second(self.dropout(self.relu(out)))
        return out.view(users_embeddings.size(0), items_embeddings.size(0))


class MemoryModule(nn.Module):
    def __init__(self, num_users, num_items, memory_dim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.users_memory_backup = nn.Parameter(torch.zeros((num_users, memory_dim)), requires_grad=False)
        self.items_memory_backup = nn.Parameter(torch.zeros((num_items, memory_dim)), requires_grad=False)
        self.users_memory = None
        self.items_memory = None
        self.restore_memory_from_backup()

    def restore_memory_from_backup(self):
        self.users_memory = self.users_memory_backup.data.clone()
        self.items_memory = self.items_memory_backup.data.clone()

    def set_memory(self, users_ids, users_memory):
        self.users_memory[users_ids] = users_memory

    def update_memory(self, items_ids, items_memory):
        self.items_memory[items_ids] = (1 - self.alpha) * self.items_memory[items_ids] \
                                       + self.alpha * items_memory

    @torch.no_grad()
    def update_backup(self, users_ids, users_memory, items_ids, items_memory):
        if self.training:
            if users_ids is not None:
                self.users_memory_backup[users_ids] = (1 - self.alpha) * self.users_memory_backup[users_ids] \
                                                      + self.alpha * users_memory
            if items_ids is not None:
                self.items_memory_backup[items_ids] = (1 - self.alpha) * self.items_memory_backup[items_ids] \
                                                      + self.alpha * items_memory
