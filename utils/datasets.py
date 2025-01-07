# suppress warnings
import warnings
# Suppress specific UserWarning related to InMemoryDataset
warnings.simplefilter("ignore", category=UserWarning)

import torch
import torch_geometric
import torch_geometric.datasets as tgdatasets

DATASET_NAMES = ['Cora', 'CiteSeer', 'PubMed']
DATA_DIR = "./data/"
DATASETS = { name: tgdatasets.Planetoid(
                    root=DATA_DIR,
                    name=name,
                    split="public",
                    transform=torch_geometric.transforms.GCNNorm()
                    ) 
            for name in DATASET_NAMES
            }


def mask_size(mask):
    return torch.count_nonzero(mask).item()

def print_datasets_info():
    for name, dataset in DATASETS.items():
        print(f"Info for {name}:")
        print(f"num_nodes: {dataset.data.x.shape[0]}")
        print(f"num_edges: {dataset.data.edge_index.shape[1]}")
        print(f"num_node_features = {dataset.num_node_features}")
        print(f"num_classes = {dataset.num_classes}")
        
        print(f"Training set size: {mask_size(dataset.data.train_mask)}")
        print(f"Validation set size: {mask_size(dataset.data.val_mask)}")
        print(f"Test set size: {mask_size(dataset.data.test_mask)}")
        print()

def load_data(dataset_name, data_only=False):
    """
    Returns the dataset and [Data] object for the given dataset name.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = DATASETS[dataset_name]
    data = dataset.data.to(device)
    return data if data_only else (data, dataset)