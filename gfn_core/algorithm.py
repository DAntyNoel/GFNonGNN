import torch
import torch.nn.functional as F
import numpy as np
from torch_sparse import SparseTensor
from .utils import get_num_edges

def get_degree(s_, edge_index):
    """
    获取第 s_ 条边的 source_node 的入度个数。用于Pb的计算

    参数:
        s_ (int): 边的索引。
        edge_index (torch.Tensor or torch_sparse.SparseTensor): 边索引张量或稀疏张量。

    返回:
        int: source_node 的入度个数。
    """
    if isinstance(edge_index, torch.Tensor):
        if s_ == edge_index.size(1):
            # The terminal edge
            return edge_index.size(1)
        source_node = edge_index[0, s_]
        in_degree = (edge_index[1] == source_node).sum().item()
        return in_degree
    elif isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        if s_ == row.size(0):
            # The terminal edge
            return row.size(0)
        source_node = row[s_]
        in_degree = (col == source_node).sum().item()
    else:
        raise ValueError("edge_index must be torch.Tensor or torch_sparse.SparseTensor")
    
def init_state(batch_size, gfn_num_nodes, device):
    state = torch.zeros(batch_size, gfn_num_nodes, dtype=torch.long).to(device)
    done = torch.full((batch_size,), False, dtype=torch.bool, device=device)
    return state, done

def is_decided(state):
    """
    检查状态是否已决定。"""
    return state == 1

def sample_from_pf_logits(pf_logits, pf_stop, state, done, rand_prob=0.):
    # TODO: checkout
    # use -1 to denote impossible action (e.g. for done graphs)
    action = torch.full((pf_logits.size(0),), -1, dtype=torch.long, device='cpu')
    pf_logits[is_decided(state)] = -np.inf
    pf_logits = torch.cat([pf_logits, pf_stop], dim=1)
    pf_undone = pf_logits[~done].softmax(dim=1)
    action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1)

    return action

def step(state, done, action, edge_index, edge_weight=None):
    '''
    action: (batch_size,) torch.long
    - -1: invalid action, used when done
    - 0~state.size(1)-1: valid action
    - state.size(1): terminal action, will set done = True
    '''
    # TODO: checkout
    new_state = state.clone().to(device=state.device)
    new_done = done.clone().to(device=state.device)
    batch_size = state.size(0)
    new_done[action == state.size(1)] = True
    new_state[torch.arange(batch_size, device=state.device)[~new_done], action[~new_done]] = 1
    return new_state, new_done

def edge_similarity_mask(x, edge_index, state, threshold=0.5):
    """
    计算边的顶点特征相似度，并返回满足相似度阈值的边的mask。

    参数：
    - x: 节点特征矩阵，形状为 (num_nodes, num_features)。
    - edge_index: 边索引，形状为 (2, num_edges)。
    - state: 节点掩码，形状为 (num_nodes,)，表示哪些节点被选中。
    - threshold: 相似度阈值，范围为 [0, 1]。

    返回：
    - mask: 满足相似度阈值的边的mask，形状为 (num_edges,)。
    """
    if isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        src_features = x[row]
        dst_features = x[col]

        src_in_state = state[row].bool()
        dst_in_state = state[col].bool()
        both_in_state = src_in_state & dst_in_state
    else:
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]

        src_in_state = state[edge_index[0]].bool()
        dst_in_state = state[edge_index[1]].bool()
        both_in_state = src_in_state & dst_in_state
    similarity = F.cosine_similarity(src_features, dst_features, dim=1)
    mask = (similarity > threshold) & both_in_state

    return mask

def edge_dissimilarity_mask(x, edge_index, state, threshold=1.0):
    """
    计算边的顶点特征不相似度，并返回满足不相似度阈值的边的mask。

    参数：
    - x: 节点特征矩阵，形状为 (num_nodes, num_features)。
    - edge_index: 边索引，形状为 (2, num_edges)。
    - state: 节点掩码，形状为 (num_nodes,)，表示哪些节点被选中。
    - threshold: 不相似度阈值，范围为 [0, +∞)。

    返回：
    - mask: 满足不相似度阈值的边的mask，形状为 (num_edges,)。
    """
    if isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        src_features = x[row]
        dst_features = x[col]

        src_in_state = state[row].bool()
        dst_in_state = state[col].bool()
        both_in_state = src_in_state & dst_in_state
    else:
        src_features = x[edge_index[0]]
        dst_features = x[edge_index[1]]

        src_in_state = state[edge_index[0]].bool()
        dst_in_state = state[edge_index[1]].bool()
        both_in_state = src_in_state & dst_in_state

    distance = torch.norm(src_features - dst_features, p=2, dim=1)
    mask = (distance > threshold) & both_in_state

    return mask

def select_edge_algo(x, edge_index, rollout_batch=None, state:torch.Tensor=None, edge_weight=None):
    if state is None:
        if rollout_batch is None:
            raise ValueError("rollout_batch must be provided if state is not given")
        traj_s = rollout_batch['state'] # (batch_size, gfn_num_nodes, max_traj_len)
        traj_a = rollout_batch['action'] # (batch_size, max_traj_len)
        traj_r = rollout_batch['logr'] # (batch_size, max_traj_len)
        traj_d = rollout_batch['done'] # (batch_size, max_traj_len)
        len = rollout_batch['len'] # (batch_size,)
        # get the last logr for each batch
        traj_r = traj_r[torch.arange(traj_r.size(0)), len - 1]
        # get the highest logr batch id
        b_idx = torch.argmax(traj_r, dim=0)
        state = traj_s[b_idx, :, len[b_idx] - 1]
    else:
        if state.dim() == 2:
            state = state[0]
    # # test code
    # print(f"select_edge_algo: state shape={state.shape}")
    assert state.size(0) == x.size(0), \
        f"state size {state.size(0)} != num node {x.size(0)}"
    mask = edge_dissimilarity_mask(x, edge_index, state, threshold=0.5)
    # remove the edge index of the selected edge
    if isinstance(edge_index, torch.Tensor):
        new_edge_index = edge_index[:, mask]
    elif isinstance(edge_index, SparseTensor):
        row, col, value = edge_index.coo()
        new_row = row[mask]
        new_col = col[mask]
        new_value = value[mask] if value is not None else None
        new_edge_index = SparseTensor(row=new_row, col=new_col, value=new_value, sparse_sizes=edge_index.sparse_sizes())
    else:
        raise ValueError("edge_index must be torch.Tensor or torch_sparse.SparseTensor")
    return new_edge_index
