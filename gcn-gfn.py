import os
import os.path as osp
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log

from base_models import GCN

from utils import get_logger, Argument
from network import get_degree
from gfn import EdgeSelector

def get_dataset(params:Argument):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
    dataset = Planetoid(path, params.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(params.device)

    if params.use_gdc:
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)
    
    params.in_channels = dataset.num_features
    params.out_channels = dataset.num_classes

    return data, params

def get_models(params:Argument, data):
    model_gnn = GCN(params).to(params.device)
    optimizer = torch.optim.Adam([
        dict(params=model_gnn.convs.parameters(), weight_decay=5e-4),
        dict(params=model_gnn.out_conv.parameters(), weight_decay=0)
    ], lr=params.lr)  # Perform weight-decay except last convolution.

    best_gnn_model_path = osp.join(params.save_path, 'best_gnn_model.pt')
    best_gfn_model_path = osp.join(params.save_path, 'best_gfn_model.pt')
    torch.save(model_gnn.state_dict(), best_gnn_model_path)

    params.in_channels = data.x.size(1)
    params.out_channels = data.y.max().item() + 1
    params.best_gnn_model_path = best_gnn_model_path
    params.best_gfn_model_path = best_gfn_model_path
    params.max_degree = get_degree(data.edge_index.cpu(), data.num_nodes).max().item()
    params.num_edges = data.edge_index.size(1)
    if params.use_gfn:
        GFN = EdgeSelector(params, params.device)
        GFN.set_evaluate_tools(
            model_gnn, F.cross_entropy, data.x, data.y, data.train_mask
        )
    else:
        GFN = None
    return params, model_gnn, optimizer, GFN

def save_models(gnn_model, GFN:EdgeSelector, params:Argument, epoch=-1):
    if epoch < 0:
        gnn_model_path = osp.join(params.save_path, 'gnn_model.pt')
        gfn_model_path = osp.join(params.save_path, 'gfn_model.pt')
    else:
        gnn_model_path = osp.join(params.save_path, 'ckpt', f'Epoch_{epoch}_gnn_model.pt')
        gfn_model_path = osp.join(params.save_path, 'ckpt', f'Epoch_{epoch}_gfn_model.pt')
        os.makedirs(osp.join(params.save_path, 'ckpt'), exist_ok=True)
    torch.save(gnn_model.state_dict(), gnn_model_path)
    if GFN is not None:
        torch.save(GFN.state_dict(), gfn_model_path)
    logger.info(f'Saved models at {params.save_path}!')

def run(args:Argument):
    data, params = get_dataset(args)
    params, model_gnn, optimizer, GFN = get_models(params, data)
    if params.wandb:
        init_wandb(name=f'GCN-{params.dataset}', epochs=params.epochs,
                   hidden_channels=params.hidden_channels, lr=params.lr, device=params.device)
    if params.task_name:
        params.save(osp.join(params.save_path, 'params.json'))
    logger.info(f'Device: {params.device}')
    best_val_acc = test_acc = 0
    gfn_train_cnt = 0
    times = []
    for epoch in range(1, params.epochs + 1):
        start = time.time()
        # train
        model_gnn.train()
        optimizer.zero_grad()
        out = model_gnn(data.x, data.edge_index, GFN)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # test
        with torch.no_grad():
            model_gnn.eval()
            pred = model_gnn(data.x, data.edge_index, GFN).argmax(dim=-1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        train_acc, val_acc, tmp_test_acc = accs

        if val_acc > best_val_acc:
            logger.info(f'Best val at epoch {epoch}. Saved model!')
            torch.save(model_gnn.state_dict(), params.best_gnn_model_path)
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            if GFN is not None:
                torch.save(GFN.state_dict(), params.best_gfn_model_path)

        logger.info(
            f'Epoch: {epoch:03d},\n'
            f'Loss: {float(loss):.4f},\n'
            f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f},\n'
            f'Best Val: {best_val_acc:.4f}, Test: {test_acc:.4f}'
        )
        times.append(time.time() - start)
        if params.use_gfn and epoch % params.gfn_train_interval == 0:
            logger.info(f'GFN train at epoch {epoch}!')
            GFN.set_evaluate_tools(
                params.best_gnn_model_path, F.cross_entropy, data.x, data.y, data.train_mask
            )
            logg_gfn_ls = []
            time_gfn_ls = []
            
            for train_step in range(1, params.gfn_train_steps+1):
                start_time = time.time()
                gfn_train_cnt += 1
                loss_gfn = GFN.train_gfn()
                logger.debug(
                    f'Step: {train_step}, Total: {gfn_train_cnt}\n'
                    f'GFN_Loss: {loss_gfn:.4f}'
                )
                logg_gfn_ls.append(loss_gfn)
                time_gfn_ls.append(time.time() - start_time)
            logg_gfn_ls = torch.tensor(logg_gfn_ls)
            logger.info(
                f"GFN train done! Trained steps: {params.gfn_train_steps}. Total train steps: {gfn_train_cnt}\n"
                f"GFN Loss: From {logg_gfn_ls[0]:.4f} to {logg_gfn_ls[-1]:.4f}. Min: {logg_gfn_ls.min().item():.4f}. Avg: {logg_gfn_ls.mean().item():.4f}\n"
                f"GFN Time: {sum(time_gfn_ls):.4f}s. Median time per epoch: {torch.tensor(time_gfn_ls).median():.4f}s"
            )
        if params.save_interval > 0 and epoch % params.save_interval == 0:
            save_models(model_gnn, GFN, params, epoch)

    if params.save_interval >= 0:
        save_models(model_gnn, GFN, params)

    print(f'Total time: {sum(times):.4f}s')
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

if __name__ == '__main__':
    args = Argument().parse_args()
    if args.task_name != '':
        # setup task env
        save_path = osp.join(args.save_path, args.task_name)
        if os.path.exists(save_path):
            if args.overwrite:
                shutil.rmtree(save_path)
            else:
                raise ValueError(f'Task folder {save_path} already exists. Use --overwrite to overwrite.')
        os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
        logger = get_logger('GCN', task_folder=args.save_path)
    else:
        logger = get_logger('GCN')
    run(args)
