import torch
from torch_sparse import SparseTensor

def get_num_edges(edge_index):
    if isinstance(edge_index, torch.Tensor):
        return edge_index.size(1)
    elif isinstance(edge_index, SparseTensor):
        return edge_index.nnz()
    else:
        raise ValueError("edge_index must be torch.Tensor or torch_sparse.SparseTensor")
    