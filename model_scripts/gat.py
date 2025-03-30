import torch.nn.functional as F
import torch
from torch.nn import Linear, ParameterList, Module
from torch_geometric.nn import GATConv

class GAT(Module):
    def __init__(self, in_dim, out_dim,**layer_paras):
        super().__init__()
        self.hidden_dim = layer_paras.pop("hidden_size")

        self.conv_layers = []
        self.conv_layers +=[GATConv(in_dim, self.hidden_dim, heads=4)]
        self.conv_layers +=[GATConv(self.hidden_dim, self.hidden_dim, heads=4)]

        self.conv_layers = ParameterList(self.conv_layers)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim//2)
        self.mlp2 = Linear(self.hidden_dim//2, out_dim)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)

        return x