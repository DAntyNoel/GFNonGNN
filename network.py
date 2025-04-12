import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from utils import get_logger

def get_degree(edge_index, num_nodes):
    '''
    Get the degree of each node in the graph
    Args:
        edge_index: (2, num_edges)
        num_nodes: int
    Returns:
        degree: (num_nodes,)
    '''
    degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
    degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))

    return degree

logger_GATGFN = get_logger('network', folder='logs')

class GATGFN(torch.nn.Module):
    def __init__(self, params, graph_level_output=0):
        super().__init__()

        self.hidden_dim = params.gfn_hidden_dim
        self.num_layers = params.gfn_num_layers
        self.heads = params.gfn_heads
        self.dropout = params.gfn_dropout
        self.graph_level_output = graph_level_output

        self.input_embedding = nn.Embedding(params.max_degree+1, self.hidden_dim)
        self.inp_transform = nn.Identity()

        self.policy_model = nn.ModuleList()
        for i in range(self.num_layers):
            self.policy_model.append(GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                dropout=self.dropout,
                add_self_loops=False,
                concat=False,
            ))

        if self.graph_level_output > 0:
            self.read_out_flow = nn.Sequential(
                nn.Linear(self.num_layers * self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.graph_level_output)
            )
        
        if self.hidden_dim == self.heads * self.num_layers + 1:
            self.x_to_alpha = nn.Identity()
        else:
            self.x_to_alpha = nn.Linear(self.num_layers * self.hidden_dim, self.num_layers * self.heads + 1)
        self.read_out_logits = nn.Sequential(
            nn.Linear(self.num_layers * self.heads + 1, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state:torch.Tensor, edge_index:torch.Tensor):
        '''
        Forward pass of the GAT-based GFN model. All the batch should be the same graph.
        Args:
            state: (batch_size, num_edges)
            edge_index: (2, num_edges)
        Returns:
            logits: (batch_size, num_edges+1)
            flows: (batch_size, )
        '''
        b, e = state.size()
        x = get_degree(edge_index, edge_index.max().item()+1).to(edge_index.device) # (num_nodes,)
        x = self.input_embedding(x) # (num_nodes, hidden_dim)
        x = self.inp_transform(x) # (num_nodes, hidden_dim)
        x_ls, alpha_ls = [], []
        for layer in self.policy_model:
            x = F.dropout(x, p=0.6, training=self.training)
            x, (edge_index, alpha) = layer(x, edge_index, return_attention_weights=True)
            # alpha: (total_num_edges, heads)
            x_ls.append(x)
            alpha_ls.append(alpha)
            x = F.leaky_relu(x, negative_slope=0.2)
        
        x_ls = torch.cat(x_ls, dim=1) # (num_nodes, num_layers * hidden_dim)
        alpha_ls = torch.cat(alpha_ls, dim=1) # (num_edges, num_layers * heads)

        # expand alpha_ls batch_size times and concate with state
        alpha_ls_expanded = alpha_ls.unsqueeze(0).repeat(b, 1, 1)  # (batch_size, num_edges, num_layers * heads)
        state_expanded = state.unsqueeze(2)  # (batch_size, num_edges, 1)
        concated = torch.cat((state_expanded, alpha_ls_expanded), dim=-1)  # (batch_size, num_edges, num_layers * heads + 1)

        x_logits = self.x_to_alpha(x_ls)  # (num_nodes, num_layers * heads + 1)
        x_logits_expanded = x_logits.unsqueeze(0).repeat(b, 1, 1)  # (batch_size, num_nodes, num_layers * heads + 1)
        x_logits_expanded = x_logits_expanded.mean(dim=1).reshape(b, 1, -1)  # (batch_size, 1, num_layers * heads + 1)

        # concatenate x_logits_expanded as the TERMINAL state action logit
        logits = torch.concat((x_logits_expanded, concated), dim=1)  # (batch_size, num_edges + 1, num_layers * heads + 1)
        pf_logits = self.read_out_logits(logits)  # (batch_size, num_edges + 1,)

        if self.graph_level_output > 0:
            flows = self.read_out_flow(x_ls)
            return pf_logits, flows
        
        return pf_logits 

    def action(self, state, done, edge_index):
        '''
        Sample actions using the policy model
        Args:
            state: (batch_size, num_edges)
            done: (batch_size,)
            edge_index: (2, num_edges)
        Returns:
            action: (batch_size,)
        '''
        self.eval()
        b, e = state.size()
        with torch.no_grad():
            action = torch.full((b,), -1, dtype=torch.long, device=state.device)
            pf_logits = self(state, edge_index)
            if self.graph_level_output > 0:
                pf_logits = pf_logits[0]
            pf_logits = pf_logits.reshape(b, -1)
            state = torch.cat([state, done.unsqueeze(-1)], dim=1) # (batch_size, num_edges+1)
            pf_logits[state == 1] = -np.inf
            pf_undone = pf_logits[~done].softmax(dim=1)
            action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)
            return action
