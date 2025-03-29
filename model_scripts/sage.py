import torch.nn.functional as F
import torch
from torch.nn import Linear, ParameterList, Module
from torch_geometric.nn import SAGEConv

class SAGE(Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pass