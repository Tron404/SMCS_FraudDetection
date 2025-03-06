import torch
import pandas as pd
import os

from torch_geometric.data import Data
from torch_geometric.datasets import DGraphFin
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

from typing import Literal

### CREATE GRAPH

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_node(data: pd.DataFrame, **kwargs):
    mapping_idx = {node_address: idx for idx, node_address in enumerate(data.address.unique())}
                   
    x = torch.as_tensor(data.drop("address", axis=1).to_numpy(), dtype=torch.float32)

    return x, mapping_idx

def make_edge(data, address_mapping):
    edge_index = torch.as_tensor([[address_mapping[input_node], address_mapping[output_node]] for input_node, output_node in zip(data["input_address"], data["output_address"]) if input_node in address_mapping.keys() and output_node in address_mapping.keys()])

    return edge_index.T

def make_masks(graph: Data, labels_to_remove: list[int]=None):
    node_split = RandomNodeSplit(num_val=0.1, num_test=0.5)
    graph = node_split(graph)

    if labels_to_remove:
        assert len(set(graph.y.tolist())) - len(labels_to_remove) == 2, f"Expected {len(set(graph.y.tolist()))-2} labels, but got {len(labels_to_remove)} - necessary for binary classification"
        for label_to_remove in labels_to_remove:
            graph.train_mask[torch.where(graph.y == label_to_remove)] = False # remove unknown-class nodes from mask
            graph.test_mask[torch.where(graph.y == label_to_remove)] = False 
            graph.val_mask[torch.where(graph.y == label_to_remove)] = False 
    
    return graph

def make_graph(feature_df, edge_df, y_labels, save_to_disk=False, file_name=None):
    x_features, node_mapping = make_node(feature_df)
    edge_index = make_edge(edge_df, node_mapping)

    graph = Data()
    graph.x = x_features
    graph.y = y_labels
    graph.edge_index = edge_index

    if save_to_disk:
        if file_name is None:
            print("A file name must be provided as well.")
            raise ValueError

        torch.save(graph, file_name + ".pt")

    return graph

### LOAD DATASETS

def load_dgraphfin(path_to_folder="."):
    NUM_CLASSES = 2
    dataset = DGraphFin(root=os.path.join(path_to_folder, "dataset"))[0].to(DEVICE)
    dataset.pop("edge_type")
    dataset.pop("edge_time") # @TODO: check how the edges change over time (if at all?)
    dataset.x = normalize_column(dataset.x)

    return dataset, NUM_CLASSES

def load_create_ellipticpp(path_to_folder=".", save_to_disk=False, file_name=None):
    NUM_CLASSES = 2
    addr2addr = os.path.join(path_to_folder, "Elliptic++Dataset/AddrAddr_edgelist.csv")
    wallets_features_classes = os.path.join(path_to_folder, "Elliptic++Dataset/wallets_features_classes_combined.csv")

    addr2addr_df = pd.read_csv(addr2addr)
    wallets_features_classes_df = pd.read_csv(wallets_features_classes)
    wallets_features_classes_df = wallets_features_classes_df[
        wallets_features_classes_df["Time step"] == 1
    ]
    y_label_wallet = torch.as_tensor(wallets_features_classes_df["class"].tolist())
    y_label_wallet -= 1 # pytorch expects class labels in the range of [0, num_classes-1]
    wallets_features_classes_df = wallets_features_classes_df.drop(["class", "Time step"], axis=1)
    wallets_features_classes_df.reset_index(drop=True, inplace=True)

    dataset = make_graph(wallets_features_classes_df, addr2addr_df, y_label_wallet, save_to_disk=save_to_disk, file_name=file_name).to(DEVICE)
    dataset = make_masks(dataset, labels_to_remove=[2])

    return dataset, NUM_CLASSES

### PROCESS DATASETS

def normalize_column(data_feature, norm_type: Literal["z-score", "min-max"]="z-score"):
    def z_scoring(x: torch.Tensor):
        return (x - x.mean(dim=0))/x.std(dim=0)
    
    def minmax_scoring(x: torch.Tensor):
        return (x - x.min(dim=0))/(x.max(dim=0) - x.min(dim=0))

    match norm_type:
        case "z-score":
            data_feature = z_scoring(data_feature)
        case "min-max":
            data_feature = minmax_scoring(data_feature)
        case _:
            print("No normalization method with that name was found")
            raise NotImplemented

    return data_feature

def split_into_batches(graph: Data, num_batches, num_neighbours, num_hops, shuffle=False):
    loader = NeighborLoader(
        data=graph,
        num_neighbors=[num_neighbours] * num_hops,
        batch_size=num_batches,
        replace=False,
        shuffle=shuffle
    )

    return loader
