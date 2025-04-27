import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from utils import get_logger

logger = get_logger('GAT')

class GAT(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_channels = params.in_channels
        self.hidden_channels = params.hidden_channels
        self.out_channels = params.out_channels
        self.heads = params.heads
        self.num_layers = params.num_layers

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(self.in_channels, self.hidden_channels,
                             heads=self.heads, dropout=0.6))
        # TODO
        for _ in range(self.num_layers - 2):
            self.convs.append(GATConv(self.hidden_channels * self.heads, self.hidden_channels,
                                      heads=self.heads, dropout=0.6))
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.out_conv = GATConv(self.hidden_channels * self.heads, self.out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, GFN=None, start_layer=-1):
        if GFN is not None:
            edge_indexs = GFN.sample(x, edge_index, self.num_layers - 1)
        for i, conv in enumerate(self.convs):
            if GFN is not None:
                x = conv(x, edge_indexs[i])
                x = F.leaky_relu(x, negative_slope=0.2)
            else:
                x = conv(x, edge_index)
                x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=0.6, training=self.training)
            if i > start_layer > 0:
                # start_layer is the layer to train early GNN, so that
                # the GFN can be trained with the early GNN features.
                break
        x = self.out_conv(x, edge_index)
        return x
