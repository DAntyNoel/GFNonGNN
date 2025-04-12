import torch
import torch.nn.functional as F

from buffer import ReplayBufferDB
from network import GATGFN

class GFNBase(object):
    def __init__(self):
        self.gnn_model = None
        self.criterion = None
        self.x = None
        self.y = None
        self.mask = None

    # An offline model of GCN for evaluating the GFN model, used in reward_fn
    def set_evaluate_tools(self, gnn_model=None, criterion=None, x=None, y=None, mask=None):
        if gnn_model is not None:
            # path = self.params.temp_model_path
            # if not osp.exists(path):
            #     os.makedirs(path)
            # file_name = 'frozen_model.pt'
            # if not osp.exists(osp.join(path, file_name)):
            #     os.system(f"touch {osp.join(path, file_name)}")
            # path = osp.join(path, file_name)
            # gcn_model.save_dict(path)
            # frozen_gcn_model = type(gcn_model)(self.params)
            # frozen_gcn_model.load_dict(path)
            # self.gcn_model = frozen_gcn_model.eval().to(self.params.device)
            self.gnn_model = gnn_model
        if criterion is not None:
            self.criterion = criterion
        if x is not None:
            self.x = x.to(self.params.device)
        if y is not None:
            self.y = y.to(self.params.device)
        if mask is not None:
            self.mask = mask.to(self.params.device)
            
    def reward_fn(self, edge_index, state):
        edge_index = edge_index[:, state]
        loss = self.criterion(
            self.gnn_model.forward_with_fixed_gfn(self.x, edge_index)[self.mask], self.y[self.mask]
        )
        reward = torch.exp(-loss / self.params.reward_scale)
        return reward
    
    def init_state(self, batch_size, num_edges):
        # Initialize the state and done variables
        state = torch.zeros((batch_size, num_edges), dtype=torch.long)
        done = torch.zeros((batch_size,), dtype=torch.bool)
        return state, done
    
    def step(self, state:torch.Tensor, done:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Update the state and done variables based on the selected actions
        Args:
            state: (batch_size, num_edges)
            done: (batch_size,)
            action: (batch_size,)
        Returns:
            state: (batch_size, num_edges)
            done: (batch_size,)
        '''
        b, e = state.size()
        # filter valid action. 
        mask1 = action >= 0
        mask2 = action < e
        action_done = mask1 & mask2
        if self.params.check_step_action:
            action_valid = action.clone().to(action.device)
            action_valid[action_done] = 0
            # Treat invalid action as done
            mask3 = state[:, action_valid] == 1
            action_done = action_done | mask3
        done = done | action_done
        # update state
        state[done][action[done]] = 1
        return state, done
     

def get_in_degree(s_, edge_index):
    if s_ == edge_index.size(1):
        # The terminal edge
        return edge_index.size(1)
    source_node = edge_index[0, s_]
    in_degree = (edge_index[1] == source_node).sum().item()
    return in_degree

class EdgeSelector(GFNBase):
    def __init__(self, params):
        super().__init__()
        self.model_Pf = GATGFN(params)
        self.model_F = GATGFN(params, graph_level_output=1)
        self.parameters_ls = [
            self.model_Pf.parameters(),
            self.model_F.parameters(),
        ]
        if params.use_pb:
            self.model_Pb = GATGFN(params)
            self.parameters_ls.append(self.model_Pb.parameters())
        else:
            self.model_Pb = None
        self.buffer = ReplayBufferDB(params)

        self.rollout_batch_size = params.rollout_batch_size
        self.num_edges = params.num_edges

        self.train_gfn_batch_size = params.train_gfn_batch_size
        self.optimizer = torch.optim.Adam(
            self.parameters_ls,
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
        self.forward_looking = params.forward_looking
        self.leaf_coef = params.leaf_coef # Origin DB w/o forward looking

    def sample(self, edge_index):
        '''
        Sample edges using the policy model
        Args:
            edge_index: (2, num_edges)
        Returns:
            edge_index: (2, num_edges)'''
        state, done = self.init_state(self.rollout_batch_size, self.num_edges) # (rollout_batch_size, num_edges), (rollout_batch_size,)
        reward = self.reward_fn(edge_index, state) # (rollout_batch_size,)
        traj_s, traj_r, traj_a, traj_d =  [], [], [], []
        while not torch.all(done):
            # Sample actions using the policy model
            action = self.model_Pf.action(state, done, edge_index) # (rollout_batch_size,)
            # Update the state and done variables based on the selected actions
            state, done = self.step(state, done, action)
            reward = self.reward_fn(edge_index, state)
            traj_s.append(state.clone())
            traj_r.append(reward.clone())
            traj_a.append(action)
            traj_d.append(done.clone())
        # save last state
        traj_s.append(state.clone())
        traj_d.append(done.clone())
        traj_r.append(reward.detach().clone())
        
        traj_s = torch.stack(traj_s, dim=2) # (rollout_batch_size, num_edges, max_traj_len)
        """
        traj_s is the dense bool tensor form of the union of traj_a
        """
        traj_a = torch.stack(traj_a, dim=1) # (rollout_batch_size, max_traj_len-1)
        """
        traj_a is tensor like 
        [ 4, 30, 86, 95, 96, 29, -1, -1],
        [47, 60, 41, 11, 55, 64, 80, -1],
        [26, 38, 13,  5,  9, -1, -1, -1]
        """
        traj_d = torch.stack(traj_d, dim=1) # (rollout_batch_size, max_traj_len)
        """
        traj_d is tensor like 
        [ F,  F,  F,  F,  F,  F,  F,  T,  T],
        [ F,  F,  F,  F,  F,  F,  F,  F,  T],
        [ F,  F,  F,  F,  F,  F,  T,  T,  T]
        """
        traj_r = torch.stack(traj_r, dim=1) # (rollout_batch_size, max_traj_len)
        traj_len = 1 + torch.sum(~traj_d, dim=1) # (rollout_batch_size,)

        roll_out_batch = {
            'state': traj_s,
            'action': traj_a,
            'logr': traj_r,
            'done': traj_d,
            'len': traj_len,
            'edge_index': edge_index,
        }
        self.buffer.add_rollout_batch(roll_out_batch)
        
        # Select final edges
        b_idx = torch.argmax(traj_r[torch.arange(self.rollout_batch_size), traj_len - 1], dim=0)
        state = traj_s[b_idx, :, traj_len[b_idx] - 1]
        return edge_index[: state]

    def train_gfn(self):
        '''
        Train the GFN policy model using the replay buffer
        '''
        # Sample a batch of rollout batches from the replay buffer
        batch_size = self.train_gfn_batch_size
        batch = self.buffer.DB_sample_batch(batch_size)
        # s, s_next, d, a, r, r_next, edge_index = batch # Tuple form
        state = batch['s'] # (batch_size, num_edges)
        state_next = batch['s_next'] # (batch_size, num_edges)
        done = batch['done'] # (batch_size,)
        action = batch['a'] # (batch_size,)
        logr = batch['r'] # (batch_size,)
        logr_next = batch['r_next'] # (batch_size,)
        edge_index_ls = batch['edge_index'] # list of (2, num_edges)

        # Train the GFN model using the sampled batch
        # TODO serialize the edge_index
        log_pf = torch.zeros(batch_size, device=state.device)
        flows = torch.zeros(batch_size, device=state.device)
        flows_next = torch.zeros(batch_size, device=state.device)
        log_pb = torch.concatenate(
            [1 / get_in_degree(a, e) for a, e in zip(action, edge_index_ls)],
        )

        for i in range(batch_size):
            edge_index_i = edge_index_ls[i].to(state.device)

            pf_logits_i = self.model_Pf(state[i].unsqueeze(0), edge_index_i) # (1, num_edges_i+1)
            log_pf[i] = F.log_softmax(pf_logits_i, dim=1)[0, action[i]]

            if self.model_Pb is not None:
                pb_logits_i = self.model_Pb(state_next[i].unsqueeze(0), edge_index_i) # (1, num_edges_i+1)
                log_pb[i] = F.log_softmax(pb_logits_i[0], dim=1)[torch.arange(batch_size), action]

            two_states = torch.stack([state[i], state_next[i]], dim=0)
            _, flow_logits = self.model_F(two_states, edge_index_i) # (2,)
            flows[i], flows_next[i] = flow_logits[0], flow_logits[1]
            
        if self.forward_looking:
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
            loss = (losses[done].sum() * self.leaf_coef + losses[~done].sum()) / batch_size
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()