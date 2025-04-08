# @TODO: @base gnn to add counting para function for time complexity
import torch

from abc import ABC, abstractmethod
from torch.nn import Module

class BaseGNN(Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_num_parameters(self) -> int:
        total_parameter_count = 0
        for layer_parameters in self.parameters():
            aux_parameter_count = 1
            for layer_dim in layer_parameters.shape:
                aux_parameter_count *= layer_dim
            total_parameter_count += aux_parameter_count

        return total_parameter_count