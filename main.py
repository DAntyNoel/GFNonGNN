import os
import os.path as osp
import shutil
import time
import wandb
import logging

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

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
    if not params.gnn_only:
        GFN = EdgeSelector(params, params.device)
        gnn_model_frozen = GCN(params)
        gnn_model_frozen.load_state_dict(torch.load(best_gnn_model_path))
        gnn_model_frozen.to(params.device)
        GFN.set_evaluate_tools(
            gnn_model_frozen, F.cross_entropy, data.x, data.y, data.train_mask
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
    logger_main.info(f'Saved models at {params.save_path}!')

def get_next_run_name(save_path, base_name):
    existing_names = os.listdir(save_path)
    existing_indices = []
    for name in existing_names:
        if name.startswith(base_name):
            try:
                index = int(name[len(base_name)+1:])
                existing_indices.append(index)
            except ValueError:
                pass
    next_index = max(existing_indices, default=0) + 1

    return f"{base_name}-{next_index}"

def run(args:Argument, logger:logging.Logger, search_k_vs:dict={}):
    data, params = get_dataset(args)
    params, model_gnn, optimizer, GFN = get_models(params, data)
    if params.task_name:
        params.save(osp.join(params.save_path, 'params.json'))
    
    if params.wandb:
        param_dict = params.as_dict()
        if search_k_vs != {}:
            # agent.py
            for key in search_k_vs.keys():
                logger.debug(f"Key {key} is in search space. Skip wandb init config.")
                param_dict.pop(key)

        run = wandb.init(
            project=params.project_name,
            name=params.task_name,
            config=param_dict,
            dir=params.save_path,
        )

        if search_k_vs != {}:
            # agent.py
            for key in search_k_vs.keys():
                logger.debug(f"Key {key} is in search space. Set from wandb config {wandb.config[key]}.")
                setattr(params, key, wandb.config[key])

        wandb.define_metric('step_GNN', step_metric='step_GNN', hidden=True)
        wandb.define_metric('step_GFN', step_metric='step_GFN', hidden=True)
        wandb.define_metric('loss/GNN', step_metric='step_GNN')
        wandb.define_metric('loss/GFN', step_metric='step_GFN')
        wandb.define_metric('acc/*', step_metric='step_GNN')
        logger.info(f"Wandb config: {wandb.config}")
        logger.info(f"Wandb URL: {run.url}")
    else:
        logger.info(f"Run without wandb. Save path: {params.save_path}")

    logger.info(f'Device: {params.device}')
    best_val_acc = test_acc = 0
    gfn_train_cnt = bad_cnt = 0
    for epoch in range(1, params.epochs + 1):
        # train GNN
        model_gnn.train()
        optimizer.zero_grad()
        if params.gnn_early_train > 0:
            start_layer = max(1, params.num_layers * epoch // params.gnn_early_train)
        out = model_gnn(data.x, data.edge_index, GFN, start_layer)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        logger.info(
            f'Epoch: {epoch:04d}, \n'
            f'Loss: {float(loss.detach()):.4f}'
        )
        if params.wandb:
            wandb.log({'loss/GNN': float(loss), 'step_GNN': epoch})

        # train GFN
        if (not params.gnn_only 
            and epoch % params.gfn_train_interval == 0
            and epoch >= params.gnn_early_train > 0
        ):
            logger.info(f'GFN train at epoch {epoch}!')
            GFN.set_evaluate_tools(
                params.best_gnn_model_path, F.cross_entropy, data.x, data.y, data.train_mask
            )
            loss_gfn_ls = []
            time_gfn_ls = []
            
            for train_step in range(1, params.gfn_train_steps+1):
                start_time = time.time()
                gfn_train_cnt += 1
                loss_gfn = GFN.train_gfn(x=data.x)
                logger.debug(
                    f'Step: {train_step}, Total: {gfn_train_cnt}\n'
                    f'GFN_Loss: {loss_gfn:.4f}'
                )
                loss_gfn_ls.append(loss_gfn)
                time_gfn_ls.append(time.time() - start_time)

                if params.wandb:
                    wandb.log({'loss/GFN': loss_gfn, "step_GFN":gfn_train_cnt})

            loss_gfn_ls = torch.tensor(loss_gfn_ls)

            logger.info(
                f"GFN train done! Trained steps: {params.gfn_train_steps}. Total train steps: {gfn_train_cnt}\n"
                f"GFN Loss: From {loss_gfn_ls[0]:.4f} to {loss_gfn_ls[-1]:.4f}. Min: {loss_gfn_ls.min().item():.4f}. Avg: {loss_gfn_ls.mean().item():.4f}\n"
                f"GFN Time: {sum(time_gfn_ls):.4f}s. Median time per epoch: {torch.tensor(time_gfn_ls).median():.4f}s"
            )

        if epoch % params.eval_interval == 0 or epoch == params.epochs:
            # test
            with torch.no_grad():
                model_gnn.eval()
                pred = model_gnn(data.x, data.edge_index, GFN).argmax(dim=-1)
                accs = []
                for mask in [data.train_mask, data.val_mask, data.test_mask]:
                    accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            train_acc, val_acc, tmp_test_acc = accs

            if val_acc > best_val_acc:
                bad_cnt = 0
                logger.info(f'Best val at epoch {epoch}. Saved model!')
                torch.save(model_gnn.state_dict(), params.best_gnn_model_path)
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                if GFN is not None:
                    torch.save(GFN.state_dict(), params.best_gfn_model_path)
            else:
                bad_cnt += 1
                if bad_cnt > params.patience > 0:
                    logger.info(f'No improvement for {params.patience} epochs. Early stopping!')
                    break

            logger.info(
                f'Eval Accuracy: \n'
                f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f},\n'
                f'Best Val: {best_val_acc:.4f}, Test: {test_acc:.4f}'
            )

            if params.wandb:
                wandb.log({
                    'acc/train': train_acc,
                    'acc/val': val_acc,
                    'acc/test': tmp_test_acc,
                    'step_GNN': epoch
                })
            
        if params.save_interval > 0 and epoch % params.save_interval == 0:
            save_models(model_gnn, GFN, params, epoch)

    if params.save_interval >= 0:
        save_models(model_gnn, GFN, params)
    if params.wandb:
        wandb.run.summary["best_val_acc"] = best_val_acc
        wandb.finish()


if __name__ == '__main__':
    args = Argument().parse_args()
    if args.task_name != '':
        # setup task env
        args.save_path = osp.join(args.save_path, args.project_name)
        os.makedirs(args.save_path, exist_ok=True)
        args.task_name = get_next_run_name(args.save_path, args.task_name)
        save_path = osp.join(args.save_path, args.task_name)
        if os.path.exists(save_path):
            if args.overwrite:
                shutil.rmtree(save_path)
            else:
                raise ValueError(f'Task folder {save_path} already exists. Use --overwrite to overwrite.')
        os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
        logger_main = get_logger('main', main_logger_level=args.log_level.upper(), task_folder=args.save_path)
        logger_main.info(f"Task name: {args.task_name}")
    else:
        logger_main = get_logger('main', main_logger_level=args.log_level.upper())
    try:
        run(args, logger_main)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger_main.error(f"Unexpected error occurred. Check debug logs for more details.")
        logger_main.error(f"Error: {e}")
        if args.wandb:
            wandb.finish()
