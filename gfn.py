import torch
import torch.nn.functional as F

from buffer import ReplayBufferDB
from network import GATGFN
from utils import get_logger

logger = get_logger('GFN')

class GFNBase(object):
    def __init__(self, params):
        self._params = params
        self.device = params.evaluate_device
        self.check_step_action = params.check_step_action
        self.reward_scale = params.reward_scale
        self.gnn_model = None
        self.criterion = None
        self.x = None
        self.y = None
        self.mask = None

    # An offline model of GCN for evaluating the GFN model, used in reward_fn
    def set_evaluate_tools(self, gnn_model=None, criterion=None, x=None, y=None, mask=None):
        if gnn_model is not None:
            if isinstance(gnn_model, str):
                self.gnn_model.load_state_dict(torch.load(gnn_model, map_location=self.device))
            else:
                self.gnn_model = gnn_model.to(self.device)
        if criterion is not None:
            self.criterion = criterion
        if x is not None:
            self.x = x.to(self.device)
        if y is not None:
            self.y = y.to(self.device)
        if mask is not None:
            self.mask = mask.to(self.device)
            
    def reward_fn(self, edge_index, state):
        data_device = state.device
        b, e = state.size()
        x = self.x
        y = self.y
        mask = self.mask
        reward_ls = []
        # TODO 
        for i in range(b):
            state_i = state[i]
            edge_index_i = edge_index[:, state_i==0].to(self.device)
            loss = self.criterion(
                self.gnn_model(x, edge_index_i)[mask], y[mask]
            )
            reward = torch.exp(-loss / self.reward_scale) / self.reward_scale
            reward_ls.append(reward)
        
        return torch.stack(reward_ls, dim=0).to(data_device) # (batch_size,)
    
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
            action: (batch_size, num_edges+1)
        Returns:
            state: (batch_size, num_edges)
            done: (batch_size,)
        '''
        b, e = state.size()
        done = done | action[:, e]
        state[~done] = state[~done] | action[~done, :e]
        return state, done
     
def get_in_degree(s_, edge_index):
    if s_ == edge_index.size(1):
        # The terminal edge
        return edge_index.size(1)
    source_node = edge_index[0, s_]
    in_degree = (edge_index[1] == source_node).sum().item()
    return in_degree

class EdgeSelector(GFNBase):
    '''
    GFN model for edge selection
    '''
    def __init__(self, params, device):
        super().__init__(params)
        self.model_Pf = GATGFN(params).to(device)
        self.model_F = GATGFN(params, graph_level_output=1).to(device)
        self.parameters_ls = [
            list(self.model_Pf.parameters()),
            list(self.model_F.parameters()),
        ]
        if params.use_pb:
            self.model_Pb = GATGFN(params).to(device)
            self.parameters_ls.append(list(self.model_Pb.parameters()))
        else:
            self.model_Pb = None
        self.buffer = ReplayBufferDB(params)

        self.rollout_batch_size = params.rollout_batch_size
        self.num_edges = params.num_edges
        self.max_traj_len = params.max_traj_len
        self.multi_edge = params.multi_edge
        self.norm_p = params.norm_p

        self.train_gfn_batch_size = params.train_gfn_batch_size
        self.optimizer = torch.optim.Adam(
            [param for sublist in self.parameters_ls for param in sublist],
            lr=params.gfn_lr,
            weight_decay=params.gfn_weight_decay,
        )
        self.forward_looking = params.forward_looking
        self.leaf_coef = params.leaf_coef # Origin DB w/o forward looking

    @torch.no_grad()
    def sample(self, edge_index, repeats=1):
        '''
        Sample edges using the policy model
        Args:
            edge_index: (2, num_edges)
            repeats: number of samples needed
        Returns:
            edge_index: (repeats, 2, num_edges_selected)'''
        states, log_rs = [], []
        for _ in range(max(1, (repeats - 1)//self.rollout_batch_size + 1)):
            state, done = self.init_state(self.rollout_batch_size, self.num_edges) # (rollout_batch_size, num_edges), (rollout_batch_size,)
            state = state.to(edge_index.device)
            done = done.to(edge_index.device)
            reward = self.reward_fn(edge_index.clone(), state) # (rollout_batch_size,)
            traj_s, traj_r, traj_a, traj_d = [], [], [], []

            while not torch.all(done):
                # Sample actions using the policy model
                action_cnt = len(traj_s)
                if self.multi_edge:
                    action = self.model_Pf.mul_action(
                        state, done, edge_index, 
                        length_penalty=float((action_cnt-1)/self.max_traj_len)
                    ) # (rollout_batch_size, num_edges+1)
                else:
                    action = self.model_Pf.action(
                        state, done, edge_index, 
                        length_penalty=float((action_cnt-1)/self.max_traj_len)
                    ) # (rollout_batch_size, num_edges+1)
                # Update the state and done variables based on the selected actions
                state, done = self.step(state, done, action)
                if action_cnt > self.max_traj_len > 0:
                    logger.debug('Max trajectory length reached')
                    break
                reward = self.reward_fn(edge_index, state)
                traj_s.append(state.clone())
                traj_r.append(reward.clone())
                traj_a.append(action)
                traj_d.append(done.clone())
            
            # save last state
            traj_s.append(state.clone())
            traj_d.append(done.clone())
            traj_r.append(reward.detach().clone())
            logger.debug(f"sample traj_len: {len(traj_s)}")
            
            traj_s = torch.stack(traj_s, dim=2) # (rollout_batch_size, num_edges, max_traj_len)
            """
            traj_s is the dense bool tensor form of the union of traj_a
            """
            traj_a = torch.stack(traj_a, dim=1) # (rollout_batch_size, num_edges, max_traj_len-1)
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

            b_arange = torch.arange(self.rollout_batch_size, device=edge_index.device)
            log_rs.append(traj_r[b_arange, traj_len[b_arange] - 1]) # (rollout_batch_size)
            states.append(traj_s[b_arange, :, traj_len[b_arange] - 1]) # (rollout_batch_size, num_edges)

        log_rs = torch.cat(log_rs, dim=0) # (>=repeats)
        states = torch.cat(states, dim=0) # (>=repeats, num_edges)
        # Select final edges
        values, indices = log_rs.topk(repeats, dim=0, largest=True, sorted=False)
        states_fin = states[indices] # (repeats, num_edges)
        edge_index_fin = []
        for r in range(repeats):
            states = states_fin[r]
            edge_index_fin.append(edge_index[:, states==0])
        return edge_index_fin

    def train_gfn(self):
        '''
        Train the GFN policy model using the replay buffer
        '''
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()

        # Sample a batch of rollout batches from the replay buffer
        batch_size = min(len(self.buffer), self.train_gfn_batch_size)
        batch = self.buffer.DB_sample_batch(valid_size=batch_size, device=self.model_F.input_embedding.weight.device)
        # s, s_next, d, a, r, r_next, edge_index = batch # Tuple form
        state = batch['s'] # (batch_size, num_edges)
        state_next = batch['s_next'] # (batch_size, num_edges)
        done = batch['done'] # (batch_size,)
        action = batch['a'] # (batch_size, num_edges)
        logr = batch['r'] # (batch_size,)
        logr_next = batch['r_next'] # (batch_size,)
        edge_index_ls = batch['edge_index'] # list of (2, num_edges)

        # Train the GFN model using the sampled batch
        # TODO serialize the edge_index
        log_pf = torch.zeros(batch_size, device=state.device)
        flows = torch.zeros(batch_size, device=state.device)
        flows_next = torch.zeros(batch_size, device=state.device)
        if self.multi_edge:
            log_pb = -torch.sum(state, dim=1)
        else:
            log_pb = torch.tensor(
                [1 / get_in_degree(torch.nonzero(a), e) for a, e in zip(action, edge_index_ls)],
                device=state.device
            )

        for i in range(batch_size):
            edge_index_i = edge_index_ls[i].to(state.device)

            if self.multi_edge:
                _, pf_logits_i = self.model_Pf.action(state[i].unsqueeze(0), done[i].unsqueeze(0), edge_index_i, return_logits=True) # (1, num_edges_i+1)
                now_selected_mask = action[i]
                now_unselected_mask = ~now_selected_mask
                selected = torch.cat([state[i] == 1, done[i].unsqueeze(0)], dim=0)
                log_pf[i] = -(torch.sum(pf_logits_i[0, now_selected_mask & ~selected]) + torch.sum((1-pf_logits_i)[0, now_unselected_mask & ~selected]))
            else:
                pf_logits_i = self.model_Pf(state[i].unsqueeze(0), edge_index_i) # (1, num_edges_i+1)
                log_pf[i] = F.log_softmax(pf_logits_i, dim=1)[0, action[i]]

            if self.model_Pb is not None:
                raise NotImplementedError
                _, pb_logits_i = self.model_Pb.action(state_next[i].unsqueeze(0), done[i].unsqueeze(0), edge_index_i, return_logits=True) # (1, num_edges_i+1)
                log_pb[i] = torch.prod(pb_logits_i[action[i]])

            two_states = torch.stack([state[i], state_next[i]], dim=0)
            _, flow_logits = self.model_F(two_states, edge_index_i) # (2,)
            flows[i], flows_next[i] = flow_logits[0], flow_logits[1]
        
        if self.norm_p and self.multi_edge:
            log_pf = log_pf / state.shape[1]
            log_pb = log_pb / state.shape[1]
            
        if self.forward_looking:
            flows_next.masked_fill_(done, 0.)
            lhs = logr.detach() + flows + log_pf # (batch_size,)
            rhs = logr_next.detach() + flows_next + log_pb
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
        else:
            flows_next = torch.where(done, logr_next, flows_next)
            lhs = flows + log_pf
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[done].sum() * self.leaf_coef + losses[~done].sum()) / batch_size
        
        loss.backward()
        self.optimizer.step()
        del state, state_next, done, action, logr, logr_next, edge_index_ls, log_pf, flows, flows_next, log_pb
        return loss.detach().item()
    
    def state_dict(self):
        return {
            'model_Pf': self.model_Pf.state_dict(),
            'model_F': self.model_F.state_dict(),
            'model_Pb': self.model_Pb.state_dict() if self.model_Pb is not None else None,
            'optimizer': self.optimizer.state_dict(),
        }
    def load_state_dict(self, state_dict):
        self.model_Pf.load_state_dict(state_dict['model_Pf'])
        self.model_F.load_state_dict(state_dict['model_F'])
        if self.model_Pb is not None:
            self.model_Pb.load_state_dict(state_dict['model_Pb'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
