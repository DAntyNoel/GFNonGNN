import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

from utils import get_logger, Argument
from network import get_degree
from gfn import EdgeSelector

args = Argument(explicit_bool=True).parse_args()

device = torch_geometric.device('auto')
# device = 'cpu'

init_wandb(
    name=f'GCN-{args.dataset}',
    lr=args.lr,
    epochs=args.epochs,
    hidden_channels=args.hidden_channels,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)

logger = get_logger('GCN', folder='logs')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.out_conv = GCNConv(hidden_channels, out_channels,
                             normalize=not args.use_gdc)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels,
                             normalize=not args.use_gdc))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                      normalize=not args.use_gdc))

    def forward(self, x, edge_index, GFN:EdgeSelector=None):
        for conv in self.convs:
            if GFN is not None:
                edge_index_i = GFN.sample(edge_index)
                logger.debug(f'GFN sample new edge_index.shape: {edge_index_i.shape}')
                x = conv(x, edge_index_i).relu()
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.out_conv(x, edge_index)
        return x


model_gnn = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
).to(device)

best_model_path = osp.join(args.save_path, 'best_model.pt')
args.best_model_path = best_model_path

args.max_degree = get_degree(data.edge_index.cpu(), data.num_nodes).max().item()

args.num_edges = data.edge_index.size(1)

torch.save(model_gnn.state_dict(), args.best_model_path)

GFN = EdgeSelector(args, device)
GFN.set_evaluate_tools(
    model_gnn, F.cross_entropy, data.x, data.y, data.train_mask
)

optimizer = torch.optim.Adam([
    dict(params=model_gnn.convs.parameters(), weight_decay=5e-4),
    dict(params=model_gnn.out_conv.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train():
    model_gnn.train()
    optimizer.zero_grad()
    out = model_gnn(data.x, data.edge_index, GFN)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model_gnn.eval()
    pred = model_gnn(data.x, data.edge_index, GFN).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    logger.info(f"Epoch {epoch} started!")
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        logger.info(f'Best val. Saved model at epoch {epoch}!')
        torch.save(model_gnn.state_dict(), args.best_model_path)
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    logger.info(
        f'Epoch: {epoch:03d}, \n'
        f'Loss: {loss:.4f}, \n'
        f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f},\n'
        f'Best Val: {best_val_acc:.4f}, Test: {test_acc:.4f},\n'
    )
    times.append(time.time() - start)
    if epoch % args.gfn_train_interval == 0:
        logger.info(f'Epoch {epoch} gfn train started!')
        GFN.set_evaluate_tools(
            best_model_path, F.cross_entropy, data.x, data.y, data.train_mask
        )
        for train_step in range(args.gfn_train_steps):
            loss = GFN.train_gfn()
            logger.info(f'Step {train_step}, \nGFN_Loss: {loss:.4f}')
print(f'Total time: {sum(times):.4f}s')
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')