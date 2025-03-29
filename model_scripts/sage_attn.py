import torch.nn.functional as F
import torch
from torch.nn import Linear, ParameterList, Module
from torch_geometric.nn import SAGEConv, GATConv

class SAGE_ATTN(Module):
    def __init__(self, in_dim, out_dim, **layer_paras):
        super().__init__()

        # self.num_layers = layer_paras.pop("num_layers",1)
        self.hidden_dim = layer_paras.pop("hidden_size")
        # self.cached = layer_paras.pop("cached", True)

        # self.dropout = layer_paras.pop("dropout", 0.0)

        self.conv_layers = []
        self.conv_layers += [SAGEConv(in_dim, self.hidden_dim)]
        self.conv_layers += [GATConv(self.hidden_dim, self.hidden_dim, heads=1)]

        self.conv_layers = ParameterList(self.conv_layers)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim//2)
        self.mlp2 = Linear(self.hidden_dim//2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.mlp1(x)
        x = F.relu(x)
        
        x = self.mlp2(x)

        return x