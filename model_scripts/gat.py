import torch

from .base_gnn import BaseGNN
from torch.nn import Linear, ParameterList, ELU, ReLU
from torch_geometric.nn import GATConv, BatchNorm, MessagePassing

class GAT(BaseGNN):
    def __init__(self, in_dim, out_dim, **layer_paras):
        super().__init__()

        self.conv_size = layer_paras.pop("conv_size")
        self.mlp_size = layer_paras.pop("mlp_size")

        self.conv_layers = []
        self.conv_layers += [
                            GATConv(in_dim, self.conv_size, heads=4),
                            ELU()
                            ]
        self.conv_layers += [
                            GATConv(self.conv_size * 4, self.conv_size, heads=6, concat=False),
                            ELU(),
                            BatchNorm(self.conv_size),
                            ]

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv_op in self.conv_layers:
            if isinstance(conv_op, MessagePassing):
                x = conv_op(x, edge_index) # do convolution
            else:
                x = conv_op(x)

        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)

        return x