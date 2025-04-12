import torch
import random

from utils import get_logger

class ReplayBufferDB(object):
    def __init__(self, params):
        self.size = params.buffer_size
        self.max_sample_batch_size = params.max_sample_batch_size

        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.pos = 0
    
    def add(self, content):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = content
        self.pos = (self.pos + 1) % self.size
    
    @staticmethod
    def get_DB_contents(contents:list):
        s, s_n, d, a, r, r_n, e = zip(*contents)
        return {
            's': torch.stack(s, dim=0), # (batch_size, num_edges)
            's_next': torch.stack(s_n, dim=0), # (batch_size, num_edges)
            'done': torch.stack(d, dim=0), # (batch_size,)
            'a': torch.stack(a, dim=0), # (batch_size,)
            'r': torch.stack(r, dim=0), # (batch_size,)
            'r_next': torch.stack(r_n, dim=0), # (batch_size,)
            'edge_index': e, # list of (2, num_edges)
        }


    def add_rollout_batch(self, rollout_batch):
        traj_s = rollout_batch['state'] # (rollout_batch_size, num_edges, max_traj_len)
        traj_a = rollout_batch['action'] # (rollout_batch_size, max_traj_len-1)
        traj_r = rollout_batch['logr'] # (rollout_batch_size, max_traj_len)
        traj_d = rollout_batch['done'] # (rollout_batch_size, max_traj_len)
        traj_len = rollout_batch['len'] # (rollout_batch_size,)
        edge_index = rollout_batch['edge_index'] # (2, num_edges)

        b = traj_s.size(0)
        for b_idx in range(b):
            traj_len_bidx = traj_len[b_idx]
            for i in range(traj_len_bidx - 1):
                transition = (
                    traj_s[b_idx, i],    # s
                    traj_s[b_idx, i+1],  # s_next
                    traj_d[b_idx, i+1],  # d
                    traj_a[b_idx, i],    # a
                    traj_r[b_idx, i],    # r
                    traj_r[b_idx, i+1],  # r_next
                    edge_index,          # edge_index
                )
                self.add(transition)

    def DB_sample_batch(self):
        '''
        Sample a batch of rollout batches from the replay buffer
        Returns:
            batch: dict
            - s: (batch_size, num_edges)
            - s_next: (batch_size, num_edges)
            - d: (batch_size,) after transition
            - a: (batch_size,)
            - r: (batch_size,)
            - r_next: (batch_size,)
            - edge_index: list of (2, num_edges)
        '''
        valid_size = min(len(self.buffer), self.max_sample_batch_size)
        batch = random.sample(self.buffer, valid_size)
        return self.get_DB_contents(batch)
   