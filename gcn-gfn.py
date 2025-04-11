import argparse
import os
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

from torch_sparse import SparseTensor
from tap import Tap # typed-argument-parser

class Argument(Tap):
    dataset: str = 'Cora'
    hidden_channels: int = 16
    lr: float = 0.01
    epochs: int = 200
    use_gdc: bool = False
    wandb: bool = False

    # Frozen GCN-GFN model path
    temp_model_path: str = 'temp'

    # GFN parameters
    buffer_size: int = 6400 # replay buffer size
    rollout_batch_size: int = 16 # 一次rollout的batch size
    use_pb: bool = False
    batch_size: int = 64 # GFN train batch size
    gfn_hidden_dim: int = 128
    gfn_num_layers: int = 2
    gfn_train_steps: int = 10
    forward_looking: bool = True
    train_eps: bool = False # Whether to learn epsilon in GFN backbone
    gfn_dropout: float = 0.5
    leaf_coef: float = 0.1
    reward_scale: float = 1.0

    # On running parameters
    ## GCN
    device: str = 'cpu'
    in_channels: int = 1
    out_channels: int = 1
    ## GFN

args = Argument(explicit_bool=True).parse_args()

device = torch_geometric.device('auto')

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

args.device = device
args.in_channels = dataset.num_features
args.out_channels = dataset.num_classes

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

from network import GFNSample


class GCNGFN(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = GCNConv(params.in_channels, params.hidden_channels,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(params.hidden_channels, params.out_channels,
                             normalize=not args.use_gdc)
        self.gfn_model = GFNSample(params).to(params.device)

    def forward(self, x, edge_index, edge_weight=None):
        if self.training:
            self.gfn_model.set_evaluate_tools(gcn_model=self)
        edge_index1, loss_gfn_1 = self.gfn_model(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index1, edge_weight).relu()

        edge_index2, loss_gfn_2 = self.gfn_model(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index2, edge_weight)
        return x, (loss_gfn_1, loss_gfn_2)
    
    def forward_with_fixed_gfn(self, x, edge_index, edge_weight=None):
        # TODO
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def save_dict(self, path=None):
        save_dict = {
            "model": {
                "conv1": self.conv1.state_dict(),
                "conv2": self.conv2.state_dict(),
            },
            "gfn_model": self.gfn_model.save_dict(),
        }
        if path is not None:
            torch.save(save_dict, path)
            print(f"GCN-GFN model saved to {path}")
        return save_dict
    def load_dict(self, path=None, save_dict=None):
        if save_dict is None:
            if path is None:
                raise ValueError("Either path or save_dict must be provided")
            save_dict = torch.load(path)
        self.conv1.load_state_dict(save_dict["model"]["conv1"])
        self.conv2.load_state_dict(save_dict["model"]["conv2"])
        self.gfn_model.load_dict(save_dict=save_dict["gfn_model"])
        print(f"GCN-GFN model loaded from {path}")

model = GCNGFN(args).to(device)

criterion=F.cross_entropy
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.

model.gfn_model.set_evaluate_tools(
    gcn_model=model,
    criterion=criterion,
    x=data.x,
    y=data.y,
    mask=data.train_mask,
)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')