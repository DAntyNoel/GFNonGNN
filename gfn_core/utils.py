import torch
from torch_sparse import SparseTensor

def get_num_edges(edge_index):
    if isinstance(edge_index, torch.Tensor):
        return edge_index.size(1)
    elif isinstance(edge_index, SparseTensor):
        return edge_index.nnz()
    else:
        raise ValueError("edge_index must be torch.Tensor or torch_sparse.SparseTensor")
    
def get_pf_logits_from_alpha(A, alpha):
    """
    Get the logits from alpha.

    Args:
        A: adjacency matrix (adj) or SparseTensor (edge_index)
        alpha: alpha value
    Returns:
        logits: logits represents the edge weights
    """
    if isinstance(A, torch.Tensor):
        logits = A * alpha
    elif isinstance(A, SparseTensor):
        logits = A * alpha
    else:
        raise ValueError("A must be torch.Tensor or torch_sparse.SparseTensor")
    
    return logits