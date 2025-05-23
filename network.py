import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

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

logger_GATGFN = get_logger('network')

class GATGFN(torch.nn.Module):
    def __init__(self, params, graph_level_output=0):
        super().__init__()
        self._params = params
        self.hidden_dim = params.gfn_hidden_dim
        self.num_layers = params.gfn_num_layers
        self.heads = params.gfn_heads
        self.dropout = params.gfn_dropout
        self.graph_level_output = graph_level_output
        self.feature_init = params.feature_init

        if self.feature_init:
            self.input_embedding = nn.Linear(params.in_channels, self.hidden_dim)
        else:
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
        
        self.x_to_alpha = nn.Linear(self.num_layers * self.hidden_dim, self.num_layers * self.heads + 1)
        self.read_out_logits = nn.Sequential(
            nn.Linear(self.num_layers * self.heads + 1, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state:torch.Tensor, edge_index:torch.Tensor, x:torch.Tensor=None):
        '''
        Forward pass of the GAT-based GFN model. All the batch should be the same graph.
        Args:
            state: (batch_size, num_edges)
            edge_index: (2, num_edges)
            x: (num_nodes, hidden_dim), required when feature_init = True
        Returns:
            logits: (batch_size, num_edges+1)
            flows: (batch_size, )
        '''
        b, e = state.size()
        if self.feature_init:
            assert x is not None, 'x should be provided when feature_init = True'
        else:
            x = get_degree(edge_index, edge_index.max().item()+1).to(edge_index.device) # (num_nodes,)
        x = self.input_embedding(x) # (num_nodes, hidden_dim)
        x = self.inp_transform(x) # (num_nodes, hidden_dim)
        x_ls, alpha_ls = [], []
        for layer in self.policy_model:
            # x = F.dropout(x, p=0.6, training=self.training)
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
            # flows = F.relu(flows)
            return pf_logits, flows
        
        return pf_logits 

    def action(self, state, done, edge_index, x=None, length_penalty=0., return_logits=False):
        '''
        Sample actions using the policy model. If length_penalty -> 1, the action will tend to sample done.
        Args:
            state: (batch_size, num_edges)
            done: (batch_size,)
            edge_index: (2, num_edges)
            x: (num_nodes, hidden_dim), required when feature_init = True
            length_penalty: float,
            return_logits: bool, whether to return the logits
        Returns:
            action: bool, (batch_size, num_edges+1)
        '''
        self.eval()
        b, e = state.size()
        if self.feature_init:
            assert x is not None, 'x should be provided when feature_init = True'
        with torch.no_grad():
            action = torch.full((b, e+1), False, dtype=torch.bool, device=state.device)
            pf_logits = self(state, edge_index, x)
            if self.graph_level_output > 0:
                pf_logits = pf_logits[0]
            pf_logits = pf_logits.reshape(b, -1)
            state = torch.cat([state, done.unsqueeze(-1)], dim=1) # (batch_size, num_edges+1)
            pf_logits[state == 1] = -np.inf
            pf_undone = pf_logits[~done].softmax(dim=1) # (batch_size, num_edges+1)
            pf_undone[:, -1] = pf_undone[:, -1] + length_penalty if length_penalty > 0. else 0
            pf_undone[:, :-1] = pf_undone[:, :-1] * (1. - length_penalty)
            sampled_indices = torch.multinomial(pf_undone, num_samples=1) # (batch_size, 1)
            action[~done] = action[~done].scatter_(dim=1, index=sampled_indices, value=1.0)
            if return_logits:
                logits = torch.zeros_like(pf_logits)
                logits[~done] = pf_undone
                return action, logits
            return action

    def mul_action(self, state, done, edge_index, x=None, length_penalty=0.):
        '''
        Sample actions using the policy model. If length_penalty -> 1, the action will tend to sample done.
        Args:
            state: (batch_size, num_edges)
            done: (batch_size,)
            edge_index: (2, num_edges)
            x: (num_nodes, hidden_dim), required when feature_init = True
            length_penalty: float,
            return_logits: bool, whether to return the logits
        Returns:
            action: bool, (batch_size, num_edges+1)
        '''
        self.eval()
        b, e = state.size()
        if self.feature_init:
            assert x is not None, 'x should be provided when feature_init = True'
        with torch.no_grad():
            action = torch.full((b, e+1), False, dtype=torch.bool, device=state.device)
            pf_logits = self(state, edge_index, x)
            if self.graph_level_output > 0:
                pf_logits = pf_logits[0]
            pf_logits = pf_logits.reshape(b, -1) # (batch_size, num_edges+1)

        min_values, _ = torch.min(pf_logits, dim=1, keepdim=True)
        max_values, _ = torch.max(pf_logits, dim=1, keepdim=True)
        normalized_logits = (pf_logits - min_values) / (max_values - min_values)
        state = torch.cat([state, done.unsqueeze(-1)], dim=1) # (batch_size, num_edges+1)
        normalized_logits[state == 1] = 0
        normalized_logits[:, e] = normalized_logits[:, e] * (1 - length_penalty) + length_penalty if length_penalty > 0. else 0
        action = Bernoulli(normalized_logits).sample() # (batch_size, num_edges+1)
        action = action.bool()
        return action
