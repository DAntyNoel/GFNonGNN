
import os
import os.path as osp
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool
from torch_sparse import SparseTensor

from gfn_core import (
    Algo,
    TransitionBuffer,
    get_num_edges,
)

DEBUG_PRINT = False  

class MLP_GIN(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        if DEBUG_PRINT:
            print(f"MLP_GIN: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        if DEBUG_PRINT:
            print(f"MLP_GIN forward: x.shape={x.shape}") 
        h = self.linears[0](x)
        if DEBUG_PRINT:
            print(f"MLP_GIN forward: h.shape={h.shape}")
        h = F.relu(self.batch_norm(h))
        return self.linears[1](h)
    
class GIN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hidden_dim = params.gfn_hidden_dim
        self.num_layers = params.gfn_num_layers
        self.output_dim = 1
        self.inp_embedding = nn.Embedding(2, params.gfn_hidden_dim)
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):  # excluding the input layer
            mlp = MLP_GIN(self.hidden_dim, self.hidden_dim, self.hidden_dim)
            self.ginlayers.append(GINConv(mlp, eps=0., train_eps=params.train_eps))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # # linear functions for graph poolings of output of each layer
        # self.linear_prediction = nn.ModuleList()
        # for layer in range(self.num_layers):
        #     self.linear_prediction.append(
        #         nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
        #         nn.Linear(self.hidden_dim, self.output_dim))
        #     )
        self.readout = nn.Sequential(
            nn.Linear(self.num_layers * self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.drop = nn.Dropout(params.gfn_dropout)
        # self.pool = # max_pool
    def forward(self, state, edge_index, edge_weight=None):
        if DEBUG_PRINT:
            print(f"GIN forward: state.shape={state.shape}")
        h = self.inp_embedding(state)
        # Reshape h to (batch_size * num_nodes, hidden_dim)
        if h.dim() == 3:
            batch_size, num_nodes, _ = h.size()
            h = h.view(batch_size * num_nodes, -1)
            graph_batch = torch.arange(batch_size, device=h.device).repeat_interleave(num_nodes)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            if DEBUG_PRINT:
                print(f"GIN layer {i}: h.shape={h.shape}")
            h = layer(h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1)) # (batch_size * num_nodes, output_dim)
        return score_over_layer.view(batch_size, -1), global_max_pool(score_over_layer, graph_batch) # (batch_size, num_nodes), (batch_size, )

class GFNSample(torch.nn.Module):
    def __init__(self, params, **evaluate_tools):
        super().__init__()
        self.buffer = TransitionBuffer(params.buffer_size)
        self.params = params
        # TODO: checkout
        self.model_Pf = GIN(params)
        self.model_F = GIN(params)
        self.model_Pb = GIN(params) if params.use_pb else None
        self.parameters_list = [
            dict(params=self.model_Pf.parameters(), weight_decay=5e-4, lr=params.lr),
            dict(params=self.model_F.parameters(), weight_decay=5e-4, lr=params.lr),
        ]
        if self.model_Pb:
            self.parameters_list.append(dict(params=self.model_Pb.parameters(), weight_decay=5e-4, lr=params.lr))
        self.optimizer = torch.optim.Adam(self.parameters_list)
        
        self.set_evaluate_tools(**evaluate_tools)
    
    def get_parameters(self):
        return list(self.model_Pf.parameters()) + list(self.model_F.parameters()) + list(self.model_Pb.parameters() if self.model_Pb else None)

    def forward(self, x, edge_index, edge_weight=None):
        '''
        x: (batch_size, num_nodes, dim_features)
        edge_index: (2, num_edges) or SparseTensor
        '''
        # TODO
        loss_gfn = None

        # rollout
        rollout_batch, metric_ls = self.rollout(x, edge_index, edge_weight)
        self.buffer.add_batch(rollout_batch)
        # select edges
        edge_index_selected = Algo.select_edge(x, edge_index, rollout_batch=rollout_batch)

        # train
        batch_size = min(len(self.buffer), self.params.batch_size)
        indices = list(range(len(self.buffer)))
        loss_ls = []
        for _ in range(self.params.gfn_train_steps):
            if len(indices) == 0 or not self.training:
                break
            print("Train GFN with frozen GCN")
            curr_indices = random.sample(indices, min(len(indices), batch_size))
            batch = self.buffer.sample_from_indices(curr_indices)
            # train step
            self.model_F.train()
            self.model_Pf.train()
            if self.model_Pb:
                self.model_Pb.train()
            torch.cuda.empty_cache()

            state = batch['state'] # (batch_size, gfn_num_nodes)
            state_next = batch['state_next'] # (batch_size, gfn_num_nodes)
            action = batch['action'] # (batch_size,)
            logr = batch['logr'] # (batch_size,)
            logr_next = batch['logr_next'] # (batch_size,)
            done = batch['done'] # (batch_size,)

            # Add terminal state
            state_embed = torch.cat([state, done.unsqueeze(1)], dim=1) # (batch_size, gfn_num_nodes+1)
            state_next_embed = torch.cat([state_next, done.unsqueeze(1)], dim=1) # (batch_size, gfn_num_nodes+1)

            two_states_embed = torch.cat([state_embed, state_next_embed], dim=0)
            pf_logits, _ = self.model_Pf(state_embed, edge_index, edge_weight)[..., 0] # (batch_size, gfn_num_nodes+1)
            _, flow_logits = self.model_F(two_states_embed, edge_index, edge_weight)[..., 0] # (2 * batch_size,)
            # pb_logits = self.model_Pb(state_next, edge_index, edge_weight)[..., 0] # (batch_size, gfn_num_nodes)
            log_pf = F.log_softmax(pf_logits, dim=1)[torch.arange(batch_size), action] # (batch_size,)
            log_pb = torch.concatenate(
                [1 / Algo.get_degree(action[i], edge_index) for i in range(batch_size)],
            )
            flows, flows_next = flow_logits[:batch_size, 0], flow_logits[batch_size:, 0]

            if self.params.forward_looking:
                flows_next.masked_fill_(done, 0.)
                lhs = logr + flows + log_pf # (batch_size,)
                rhs = logr_next + flows_next + log_pb
                loss = (lhs - rhs).pow(2)
                loss = loss.mean()
            else:
                flows_next = torch.where(done, logr_next, flows_next)
                lhs = flows + log_pf
                rhs = flows_next + log_pb
                losses = (lhs - rhs).pow(2)
                loss = (losses[done].sum() * self.params.leaf_coef + losses[~done].sum()) / batch_size
            loss_ls.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #######
        return edge_index_selected, loss_gfn

    def rollout(self, x:torch.Tensor, edge_index:torch.Tensor|SparseTensor, edge_weight=None):
        '''
        x: (num_nodes, dim_features)
        edge_index: (2, num_edges) or SparseTensor

        returns: dict
        -------
        - state: (rollout_batch_size, gfn_num_nodes, max_traj_len)
        - action: (rollout_batch_size, max_traj_len)
        - logr: (rollout_batch_size, max_traj_len)
        - done: (rollout_batch_size, max_traj_len)
        - len: (rollout_batch_size,)
        '''
        rollout_batch_size = self.params.rollout_batch_size
        gfn_num_nodes = x.size(0)
        if DEBUG_PRINT:
            print(f"GFNSample rollout: x.shape={x.shape}, edge_index.shape={edge_index.shape}")
            print(f"GFNSample rollout: gfn_num_nodes={gfn_num_nodes}")
        ## init state
        traj_s, traj_r, traj_a, traj_d = [], [], [], []
        state, done = Algo.init_state(rollout_batch_size, gfn_num_nodes, x.device)
        reward = self.reward_fn(Algo.select_edge(x, edge_index, state=state))
        while not torch.all(done):
            # sample action
            with torch.no_grad():
                self.model_Pf.eval()
                pf_logits, pf_stop = self.model_Pf(state, edge_index) # (rollout_batch_size, gfn_num_nodes+1)
                action = Algo.sample_from_pf_logits(pf_logits.cpu(), pf_stop.cpu(), state.cpu(), done.cpu(), rand_prob=0.)
                action = action.to(state.device)
            traj_s.append(state.clone())
            traj_d.append(done.clone())
            traj_r.append(reward.item())
            traj_a.append(action)
            # step with action
            state, done = Algo.step(state, done, action, edge_index)
            reward = self.reward_fn(Algo.select_edge(x, edge_index, state=state))
        ## save last state
        traj_s.append(state.clone())
        traj_d.append(done.clone())
        traj_r.append(reward.item())

        traj_s = torch.stack(traj_s, dim=2) # (rollout_batch_size, gfn_num_nodes, max_traj_len)
        """
        traj_s is the dense bool tensor form of the union of traj_a
        """
        traj_a = torch.stack(traj_a, dim=1) # (rollout_batch_size, max_traj_len)
        """
        traj_a is tensor like 
        [ 4, 30, 86, 95, 96, 29, -1, -1],
        [47, 60, 41, 11, 55, 64, 80, -1],
        [26, 38, 13,  5,  9, -1, -1, -1]
        """
        traj_d = torch.stack(traj_d, dim=1) # (rollout_batch_size, max_traj_len)
        """
        traj_d is tensor like 
        [False, False, False, False, False, False,  True,  True,  True],
        [False, False, False, False, False, False, False,  True,  True],
        [False, False, False, False, False,  True,  True,  True,  True]
        """
        traj_len = 1 + torch.sum(~traj_d, dim=1) # (rollout_batch_size,)

        batch = {
            'state': traj_s,
            'done': traj_d,
            'logr': traj_r,
            'action': traj_a,
            'len': traj_len,
        }
        return batch, {}
   
    # An offline model of GCN for evaluating the GFN model
    def set_evaluate_tools(self, gcn_model=None, criterion=None, x=None, y=None, mask=None):
        if gcn_model is not None:
            path = self.params.temp_model_path
            if not osp.exists(path):
                os.makedirs(path)
            file_name = 'frozen_model.pt'
            if not osp.exists(osp.join(path, file_name)):
                os.system(f"touch {osp.join(path, file_name)}")
            path = osp.join(path, file_name)
            gcn_model.save_dict(path)
            frozen_gcn_model = type(gcn_model)(self.params)
            frozen_gcn_model.load_dict(path)
            self.gcn_model = frozen_gcn_model.eval().to(self.params.device)
        if criterion is not None:
            self.criterion = criterion
        if x is not None:
            self.x = x.to(self.params.device)
        if y is not None:
            self.y = y.to(self.params.device)
        if mask is not None:
            self.mask = mask.to(self.params.device)
    
    def reward_fn(self, edge_index, edge_weight=None):
        loss = self.criterion(
            self.gcn_model.forward_with_fixed_gfn(self.x, edge_index, edge_weight)[self.mask], self.y[self.mask]
        )
        reward = torch.exp(-loss / self.params.reward_scale)
        return reward
         
    def save_dict(self, path=None):
        save_dict = {
            "model_Pf": self.model_Pf.state_dict(),
            "model_F": self.model_F.state_dict(),
            "model_Pb": self.model_Pb.state_dict() if self.model_Pb else None,
            "optimizer": self.optimizer.state_dict()
        }
        if path is not None:
            torch.save(save_dict, path)
            print(f"GFN model saved to {path}")
        return save_dict
    
    def load_dict(self, path=None, save_dict=None):
        if save_dict is None:
            if path is None:
                raise ValueError("Either path or save_dict must be provided")
            save_dict = torch.load(path)
        self.model_Pf.load_state_dict(save_dict["model_Pf"])
        self.model_F.load_state_dict(save_dict["model_F"])
        if self.model_Pb:
            self.model_Pb.load_state_dict(save_dict["model_Pb"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        print(f"GFN model loaded from {path}")
