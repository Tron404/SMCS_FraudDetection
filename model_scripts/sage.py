import torch
import torch.nn.functional as F

from .base_gnn import BaseGNN
from torch.nn import Linear, ParameterList, ReLU
from torch_geometric.nn import SAGEConv

class SAGE(BaseGNN):
    def __init__(self, in_dim, out_dim, **layer_paras):
        super().__init__()
        
        self.conv_size = layer_paras.pop("conv_size")
        self.mlp_size = layer_paras.pop("mlp_size")

        self.conv_layers = []

        self.conv_layers += [SAGEConv(in_dim, self.conv_size, normalize=True)]
        self.conv_layers += [SAGEConv(self.conv_size, self.conv_size, normalize=True)]

        self.conv_layers = ParameterList(self.conv_layers)

        self.mlp_layers = []
        self.mlp_layers += [
                            Linear(self.conv_size, self.mlp_size),
                            ReLU(),
                            Linear(self.mlp_size, self.mlp_size//2),
                            ReLU(),
                            Linear(self.mlp_size//2, out_dim, bias=False)
                           ]
        self.mlp_layers = ParameterList(self.mlp_layers)


        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim//2)
        self.mlp2 = Linear(self.hidden_dim//2, out_dim)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)

        return x