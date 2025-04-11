import torch
import random

DEBUG_PRINT = False

class TransitionBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.pos = 0

    def reset(self):
        self.buffer = []
        self.pos = 0

    def add_batch(self, rollout_batch):
        '''
            traj_s = rollout_batch['state'] # (batch_size, gfn_num_nodes, max_traj_len)
            traj_a = rollout_batch['action'] # (batch_size, max_traj_len)
            traj_r = rollout_batch['logr'] # (batch_size, max_traj_len)
            traj_d = rollout_batch['done'] # (batch_size, max_traj_len)
            len = rollout_batch['len'] # (batch_size,)
        '''
        traj_s = rollout_batch['state']  # (batch_size, gfn_num_nodes, max_traj_len)
        traj_a = rollout_batch['action']  # (batch_size, max_traj_len)
        traj_r = rollout_batch['logr']  # (batch_size, max_traj_len)
        traj_d = rollout_batch['done']  # (batch_size, max_traj_len)
        traj_len = rollout_batch['len']  # (batch_size,)
        
        batch_size = traj_s.shape[0]  # batch_size

        for b_idx in range(batch_size):
            traj_len_bidx = traj_len[b_idx]
            for i in range(traj_len_bidx - 1):
                transition = (
                    traj_s[b_idx, i],
                    traj_r[b_idx, i],
                    traj_a[b_idx, i],
                    traj_s[b_idx, i+1],
                    traj_r[b_idx, i+1],
                    traj_d[b_idx, i+1]
                )
                self.add_single_transition(transition)

    def add_single_transition(self, inp):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = inp
        self.pos = (self.pos + 1) % self.size

    @staticmethod
    def transition_collate_fn(transition_ls):
        s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch = \
            zip(*transition_ls)  # s_batch is a list of tensors

        s_batch = torch.cat(s_batch, dim=0)  # (sum of # nodes in batch, )
        s_next_batch = torch.cat(s_next_batch, dim=0)

        logr_batch = torch.stack(logr_batch, dim=0)
        logr_next_batch = torch.stack(logr_next_batch, dim=0)
        a_batch = torch.stack(a_batch, dim=0)
        d_batch = torch.stack(d_batch, dim=0)

        return s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch

    def sample(self, batch_size):
        # random.sample: without replacement
        batch = random.sample(self.buffer, batch_size) # list of transition tuple
        return self.transition_collate_fn(batch)

    def sample_from_indices(self, indices):
        batch = [self.buffer[i] for i in indices]
        return self.transition_collate_fn(batch)

    def __len__(self):
        return len(self.buffer)