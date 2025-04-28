import torch
import torch.nn.functional as F
from tap import Tap
import os.path as osp

from base_models import GCN, GAT
from gfn import EdgeSelector
from utils import Argument
from main import get_dataset, get_models


class Args(Tap):
    folder: str
    recur: bool = False
    
    gfn_repeats: int = 3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_states(states_fin, values, max_cols=1000):
    """
    可视化布尔张量 states_fin 为黑白格图像，并显示每个 repeats 的分数 values。
    
    参数:
        states_fin: 形状为 (repeats, num_edges) 的布尔张量。
        values: 形状为 (repeats,) 的分数向量。
        max_cols: 单个图像的最大列数，默认为 1000。
    """
    repeats, num_edges = states_fin.shape
    
    # 定义黑白颜色映射
    cmap = ListedColormap(['white', 'red'])
    
    # 将 values 转换为与 states_fin 形状相匹配的矩阵
    values_matrix = np.repeat(values[:, np.newaxis], num_edges, axis=1)
   
    num_blocks = int(np.ceil(num_edges / max_cols))
    for i in range(num_blocks):
        start_col = i * max_cols
        end_col = min((i + 1) * max_cols, num_edges)
        plt.figure(figsize=(10, 3))
        plt.imshow(states_fin[:, start_col:end_col], cmap=cmap, aspect=1)
        plt.title(f"GFN samples and reward Block {i+1}/{num_blocks} (repeats={repeats}, num_edges={end_col - start_col})")
        plt.xlabel("edges (black = selected)")
        plt.ylabel("repeats")

        plt.imshow(values_matrix[:, start_col:end_col], cmap='viridis', aspect=1, alpha=0.5)
        cbar = plt.colorbar(label='reward vs repeats', fraction=0.02, shrink=0.5)  # 调整彩色条的宽度和间距
        cbar.ax.tick_params(labelsize=8)  # 调整彩色条标签的字体大小
        print(values.min(), values.max())
        plt.show()
        break


def eval_gfn(params:Argument, args:Args):
    data, params = get_dataset(params)
    params, model_gnn, optimizer, GFN = get_models(params, data)
    assert GFN is not None, "GFN is None. Please check the model."
    # Load optimizer
    optimizer.load_state_dict(torch.load(osp.join(params.save_path, 'optimizer.pt')))
    # Load the best GNN model
    if params.best_gnn_model_path:
        # model_gnn.load_state_dict(torch.load(params.best_gnn_model_path), strict=True)
        model_gnn.load_state_dict(torch.load(osp.join(params.save_path, 'gnn_model.pt')))
        print(model_gnn.now_epoch)
        model_gnn.eval()
    else:
        raise FileNotFoundError(f"Best GNN model not found: {params.best_gnn_model_path}")
    # Load the best GFN model
    if params.best_gfn_model_path:
        GFN.load_state_dict(torch.load(osp.join(params.save_path, 'best_gfn_model.pt')))
        gnn_model_frozen = model_gnn
        GFN.set_evaluate_tools(
            gnn_model_frozen, F.cross_entropy, data.x, data.y, data.train_mask
        )
    else:
        raise FileNotFoundError(f"Best GFN model not found: {params.best_gfn_model_path}")
    
    edge_indexs, (states_fin, values, log_rs) = GFN.sample(data.x, data.edge_index, (params.num_layers - 1) * args.gfn_repeats, return_edge_logits=True)
    print(log_rs.min(), log_rs.max())
    visualize_states(states_fin.cpu().numpy(), values.cpu().numpy())

def eval_gfn_results(args:Args):
    result = torch.load(osp.join(args.folder, 'gfn_sample_result.pt'))
    states_fin = result['states_fin'].reshape(-1, result['states_fin'].shape[-1])
    values = result['values'].reshape(-1)
    visualize_states(states_fin.cpu().numpy(), values.cpu().numpy())

if __name__ == '__main__':
    args = Args().parse_args()
    if args.recur:
        params_path = osp.join(args.folder, 'params.json')
        if not osp.exists(params_path):
            raise FileNotFoundError(f"Params file not found: {params_path}")
        # Load parameters
        params = Argument().load(params_path)
        eval_gfn(params, args)
    else:
        eval_gfn_results(args)