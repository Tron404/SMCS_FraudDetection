import os
import pandas as pd
import torch

from torch_geometric.data import Data
from torch_geometric.datasets import DGraphFin
from typing import Literal, Tuple, Iterable, Optional

### CREATE GRAPH

def make_node(data: pd.DataFrame, **kwargs) -> Tuple[torch.Tensor, dict]:
    """
    Create a graph-node tensor based on some given features, and create an address->index mapping
    """
    mapping_idx = {node_address: idx for idx, node_address in enumerate(data.address.unique())}
                   
    x = torch.as_tensor(data.drop("address", axis=1).to_numpy(), dtype=torch.float32)

    return x, mapping_idx

def make_edge(data: pd.DataFrame, address_mapping: dict) -> torch.Tensor:
    """
    Create an edge between two nodes (source/input -> target/output)
    """
    edge_index = torch.as_tensor([[address_mapping[input_node], address_mapping[output_node]] for input_node, output_node in zip(data["input_address"], data["output_address"]) if input_node in address_mapping.keys() and output_node in address_mapping.keys()])

    return edge_index.T

def make_graph(feature_df: pd.DataFrame, edge_df: pd.DataFrame, y_labels: Iterable, save_to_disk: bool=False, file_name: str=None) -> Data:
    """
    Create a graph with node and edge data, and ground truth labels
    """
    x_features, node_mapping = make_node(feature_df)
    edge_index = make_edge(edge_df, node_mapping)

    if not isinstance(y_labels, torch.Tensor):
        y_labels = torch.as_tensor(y_labels)

    graph = Data()
    graph.x = x_features.contiguous()
    graph.y = y_labels.contiguous()
    graph.edge_index = edge_index.contiguous()

    if save_to_disk:
        if file_name is None:
            print("A file name must be provided as well.")
            raise ValueError

        torch.save(graph, file_name + ".pt")

    return graph

### LOAD DATASETS

def load_dgraphfin(path_to_folder: str=".") -> Tuple[Data, int]:
    """
    Load the DGraphFin dataset from disk
    """
    NUM_CLASSES = 2
    dataset = DGraphFin(root=os.path.join(path_to_folder, "dataset"))[0]
    dataset.pop("edge_type")
    # dataset.pop("edge_time") # @TODO: check how the edges change over time (if at all?)
    dataset.x = normalize_column(dataset.x, norm_type="z-score")

    return dataset, NUM_CLASSES

def load_create_ellipticpp(path_to_folder: str=".", save_to_disk: bool=False, file_name: str=None, load_test_data: bool=False, scale_data: bool=True) -> Tuple[Data, Data, Optional[Data], int]:
    """
    Load the Elliptic++ dataset from disk and create a `torch_geometric.data.Data`

    timestep can either be a singular timestep of type `int` or a range in the form of a tuple
    """
    def create_timestep_data(data_df: pd.DataFrame, address_df: pd.DataFrame, time_range: Tuple[int, int]) -> pd.DataFrame:
        timestep_left, timestep_right = time_range

        data_df = data_df[
            (data_df["Time step"] >= timestep_left) & (data_df["Time step"] <= timestep_right)
        ]

        y_label_wallet = torch.as_tensor(data_df["class"].tolist())
        data_df = data_df.drop(["class", "Time step"], axis=1)
        data_df.reset_index(drop=True, inplace=True)
        dataset = make_graph(data_df, address_df, y_label_wallet, save_to_disk=save_to_disk, file_name=file_name)

        return dataset

    NUM_CLASSES = 2
    addr2addr = os.path.join(path_to_folder, "Elliptic++Dataset/AddrAddr_edgelist.csv")
    wallets_features_classes = os.path.join(path_to_folder, "Elliptic++Dataset/wallets_features_classes_combined.csv")

    addr2addr_df = pd.read_csv(addr2addr)
    wallets_features_classes_df = pd.read_csv(wallets_features_classes)
    wallets_features_classes_df["class"] = wallets_features_classes_df["class"].replace([1,2,3],[1,0,2])
    wallets_features_classes_df = wallets_features_classes_df[wallets_features_classes_df["class"] != 2] # !!! RAMON TALKED ABOUT THIS, ALSO AUTHORS DO THIS FOR WALLET CLASSIFICATION
    
    if scale_data:
        wallets_features_classes_df.loc[:, ~wallets_features_classes_df.columns.isin(["address", "Time step", "class"])] = normalize_column(wallets_features_classes_df.loc[:, ~wallets_features_classes_df.columns.isin(["address", "Time step", "class"])].to_numpy(), norm_type="z-score").numpy()


    if load_test_data:
        train_dataset = create_timestep_data(wallets_features_classes_df, addr2addr_df, (1,37))
        test_dataset = create_timestep_data(wallets_features_classes_df, addr2addr_df, (38,42))
        return_items = train_dataset, test_dataset, NUM_CLASSES
    else:
        train_dataset = create_timestep_data(wallets_features_classes_df, addr2addr_df, (1,32))
        valid_dataset = create_timestep_data(wallets_features_classes_df, addr2addr_df, (33,37))
        return_items = train_dataset, valid_dataset, NUM_CLASSES

    return return_items

### PROCESS DATASETS

def normalize_column(data_feature: Iterable, norm_type: Literal["z-score", "min-max"]="z-score") -> torch.Tensor:
    """
    Normalize a 1D Iterable (e.g., list, or 1D tensor) based on some scoring method
    """
    def z_scoring(x: torch.Tensor):
        return (x - x.mean(dim=0))/x.std(dim=0)
    
    def minmax_scoring(x: torch.Tensor):
        return (x - x.min(dim=0))/(x.max(dim=0) - x.min(dim=0))

    if not isinstance(data_feature, torch.Tensor):
        data_feature = torch.as_tensor(data_feature)

    match norm_type:
        case "z-score":
            data_feature = z_scoring(data_feature)
        case "min-max":
            data_feature = minmax_scoring(data_feature)
        case _:
            print("No normalization method with that name was found")
            raise NotImplemented

    return data_feature
