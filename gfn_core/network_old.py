# Modified from https://github.com/zdhNarsil/GFlowNet-CombOpt/blob/main/gflownet/network.py
# Test ok
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

class MLP(nn.Module):
    """Construct two-layer MLP-type aggregator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(self, x):
        return self.linears(x)
    
class MessagePassing(nn.Module):
    def __init__(self, mlp, aggregator_type="sum", learn_eps=False):
        super().__init__()
        self.mlp = mlp
        self.aggregator_type = aggregator_type
        self.learn_eps = learn_eps
        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([0]))
        else:
            self.eps = None

    def forward(self, x, adj):
        """
        x: node features (batch_size, num_nodes, input_dim)
        adj: adjacency matrix (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
             or SparseTensor (num_nodes, num_nodes)
        """
        if isinstance(adj, SparseTensor):
            if self.aggregator_type == "sum":
                aggregated_neighs = adj.spmm(x)
            elif self.aggregator_type == "mean":
                # 对于稀疏矩阵，先计算度矩阵
                deg = adj.sum(dim=-1).clamp(min=1)
                # 归一化邻接矩阵
                adj = adj / deg.view(-1, 1)
                aggregated_neighs = adj.spmm(x)
            else:
                raise ValueError("Unknown aggregator type. Max not implemented for SparseTensor.")

        else:
            if len(adj.shape) == 2:
                adj = adj.unsqueeze(0)
            # 对于稠密矩阵，直接进行矩阵乘法
            if self.aggregator_type == "sum":
                aggregated_neighs = torch.bmm(adj, x)
            elif self.aggregator_type == "mean":
                aggregated_neighs = torch.bmm(adj, x) / (adj.sum(dim=-1, keepdim=True) + 1e-6)
            elif self.aggregator_type == "max":
                aggregated_neighs = torch.bmm(adj, x)
                aggregated_neighs = torch.max(aggregated_neighs, dim=-1, keepdim=True)[0].repeat(1, 1, x.size(-1))
            else:
                raise ValueError("Unknown aggregator type")

        if self.learn_eps:
            x = (1 + self.eps) * x + aggregated_neighs
        else:
            x = x + aggregated_neighs

        return self.mlp(x)
    
class GIN(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, num_layers=5,
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum"):
        super().__init__()

        # self.inp_embedding = nn.Embedding(input_dim, hidden_dim)
        self.inp_transform = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph_level_output = graph_level_output

        self.output_dim = output_dim

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers - 1):
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(MessagePassing(mlp, aggregator_type=aggregator_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim + graph_level_output)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, state, adj):
        """
        state: node features (batch_size, num_nodes, input_dim)
        adj: adjacency matrix (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
             or SparseTensor (num_nodes, num_nodes)
        """
        # TODO
        h = self.inp_transform(state)
        hidden_rep = [h]

        for i, layer in enumerate(self.ginlayers):
            h= layer(h, adj)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1))

        if self.graph_level_output > 0:
            graph_level_output = score_over_layer.mean(dim=1)  # Global average pooling
            return score_over_layer[..., :self.output_dim], graph_level_output[..., self.output_dim:]
        else:
            return score_over_layer
        

#####
from torch_geometric.nn import GINConv