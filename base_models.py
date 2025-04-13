import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from torch_geometric.nn import GCNConv, GATConv

from gfn import EdgeSelector
from utils import get_logger, Argument

logger = get_logger('bases')

class GCN(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_channels = params.in_channels
        self.hidden_channels = params.hidden_channels
        self.out_channels = params.out_channels
        self.use_gdc = params.use_gdc
        self.num_layers = params.num_layers

        self.out_conv = GCNConv(self.hidden_channels, self.out_channels,
                             normalize=not self.use_gdc)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_channels, self.hidden_channels,
                             normalize=not self.use_gdc))
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels,
                                      normalize=not self.use_gdc))

    def forward(self, x, edge_index, GFN:EdgeSelector=None):
        for conv in self.convs:
            if GFN is not None:
                edge_index_i = GFN.sample(edge_index)
                logger.debug(f'GFN sample new edge_index.shape: {edge_index_i.shape}')
                x = conv(x, edge_index_i).relu()
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out_conv(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_channels = params.in_channels
        self.hidden_channels = params.hidden_channels
        self.out_channels = params.out_channels
        self.heads = params.heads

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(self.in_channels, self.hidden_channels,
                             heads=self.heads, dropout=0.6))
        # TODO
        for _ in range(params.num_layers - 2):
            self.convs.append(GATConv(self.hidden_channels * self.heads, self.hidden_channels,
                                      heads=self.heads, dropout=0.6))
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.out_conv = GATConv(self.hidden_channels * self.heads, self.out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, GFN:EdgeSelector=None):
        for conv in self.convs:
            if GFN is not None:
                edge_index_i = GFN.sample(edge_index)
                logger.debug(f'GFN sample new edge_index.shape: {edge_index_i.shape}')
                x = conv(x, edge_index_i)
                x = F.leaky_relu(x, negative_slope=0.2)
            else:
                x = conv(x, edge_index)
                x = F.leaky_relu(x, negative_slope=0.2)
        x = self.out_conv(x, edge_index)
        return x
