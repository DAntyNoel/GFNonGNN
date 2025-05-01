import os
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Actor, WebKB, WikipediaNetwork

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, 'data')

def get_dataset(params):
    """
    Load the dataset.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        **kwargs: Additional arguments to pass to the dataset loader.
    
    Returns:
        Dataset: The loaded dataset.
    """
    data = None
    dataset_name:str = params.dataset
    split:int = params.split
    device = params.device
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(
            root=os.path.join(DATA_ROOT, 'Planetoid'), 
            name=dataset_name,
            split='full',
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]
        params.in_channels = dataset.num_features
        params.out_channels = dataset.num_classes
    elif dataset_name in ['Cora_geom', 'CiteSeer_geom', 'PubMed_geom']:
        dataset_name = dataset_name.split('_')[0]
        dataset = Planetoid(
            root=os.path.join(DATA_ROOT, 'Planetoid'), 
            name=dataset_name,
            split='public',
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]
        split_file = f'{dataset_name.lower()}_split_0.6_0.2_{split}.npz'
        split_path = os.path.join(DATA_ROOT, 'Planetoid', 'geom_splits', split_file)
        split_file = np.load(split_path, allow_pickle=True)
        data.train_mask = torch.Tensor(split_file['train_mask'])==1
        data.val_mask = torch.Tensor(split_file['val_mask'])==1
        data.test_mask = torch.Tensor(split_file['test_mask'])==1
        params.in_channels = dataset.num_features
        params.out_channels = dataset.num_classes
    elif dataset_name in ['Actor']:
        dataset = Actor(
            root=os.path.join(DATA_ROOT, dataset_name),
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params.in_channels = data.num_features
        params.out_channels = dataset.num_classes
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(
            root=os.path.join(DATA_ROOT, 'WikipediaNetwork'),
            geom_gcn_preprocess=True, 
            name=dataset_name,
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params.in_channels = data.num_features
        params.out_channels = dataset.num_classes
    elif dataset_name in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(
            root=os.path.join(DATA_ROOT, 'WebKB'),
            name=dataset_name,
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params.in_channels = data.num_features
        params.out_channels = dataset.num_classes
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return data.to(device), params

if __name__ == '__main__':
    from tap import Tap
    class Argument(Tap):
        dataset: str = 'Actor'
        split: int = 0
        device: str = 'cpu'
        in_channels: int = -1
        out_channels: int = -1


    params = Argument().parse_args()
    data, params = get_dataset(params)
    print(f"Dataset: {params.dataset}")
    print(f"Number of features: {params.in_channels}")
    print(f"Number of classes: {params.out_channels}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Data x.shape: {data.x.shape}")
    print(f"Data edge_index.shape: {data.edge_index.shape}")
    print(f"Number of train nodes: {data.train_mask.sum()}, {data.train_mask.shape}")
    print(f"Number of val nodes: {data.val_mask.sum()}, {data.val_mask.shape}")
    print(f"Number of test nodes: {data.test_mask.sum()}, {data.test_mask.shape}")
    print(f"Device: {params.device}")