from torch.utils.data import Sampler
import torch
from typing import Iterator, Iterable, List
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

class customGraphSampler(Sampler[List[int]]):
    def __init__(self, data: Iterable, batch_size: int=32, positive_label: int=0, negative_label: int=1, device: torch.DeviceObjType=torch.device("cpu"), rng=None) -> None:
        self.device = device
        self.rng = rng

        self.data = data
        self.batch_size = batch_size
        self.positive_label = positive_label
        self.negative_label = negative_label

        ### @TODO: check if randperm affects number of elements????

        self.y_positive_idx = torch.where(self.data.y == positive_label, 1, 0).nonzero().squeeze()
        self.y_positive_idx = self.y_positive_idx[torch.randperm(self.y_positive_idx.size()[0], generator=self.rng)]

        self.y_negative_idx = torch.where(self.data.y == negative_label, 1, 0).nonzero().squeeze()
        self.y_negative_idx = self.y_negative_idx[torch.randperm(self.y_negative_idx.size()[0], generator=self.rng)]

    # number of batches
    # @TODO: recompute based on labels
    def __len__(self) -> int:
        return self.y_positive_idx.shape[0] // (self.batch_size // 2)

    def _is_half_batch_size(self, counter_sample) -> int:
        return counter_sample >= self.batch_size // 2

    def __iter__(self) -> Iterator[int]:
        batch = []
        # torch where y == 0, y == 1 -> select first batch_size idx `y == 0` and batch_size - selection_size `y == 1`
        
        # ****@TODO: iterate over the entire dataset and add to single list? -- and discard non relevant batches
        ### get all neighbours
        # @TODO: iterate over the entire dataset but keep some other batches as well
        # @TODO: accumulate positive and negative in separate lists and then combine?
        # @TODO: sample from two lists and add indices

        # @TODO: shuffle indices?

        # print(self.y_positive_idx.shape)
        # print(self.y_negative_idx.shape)

        aux_y_positive_idx = self.y_positive_idx
        aux_y_negative_idx = self.y_negative_idx

        half_batch_size = self.batch_size // 2

        while aux_y_positive_idx.shape[0] >= half_batch_size and aux_y_negative_idx.shape[0] >= half_batch_size:
            positive_labels = aux_y_positive_idx[:half_batch_size]
            aux_y_positive_idx = aux_y_positive_idx[half_batch_size:] # ??????????????

            negative_labels = aux_y_negative_idx[:half_batch_size]
            aux_y_negative_idx = aux_y_negative_idx[half_batch_size:] # ??????????????

            batch = torch.concat([positive_labels, negative_labels], dim=-1).to(self.device).contiguous()

            yield batch

import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class customBatching:
    def __init__(self, data, positive_label: int=1, negative_label: int=0, neighbourhood_sizes: list=[10], batch_size: int=64, device: torch.DeviceObjType=torch.device("cpu")):
        self.data = data
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.neighbourhood_sizes = neighbourhood_sizes
        self.batch_size = batch_size
        self.device = device
        self.rng = torch.Generator()
        self.rng.manual_seed(42)

        torch.manual_seed(42)

        self.batch_sampler = customGraphSampler(data, batch_size=self.batch_size, positive_label=self.positive_label, negative_label=self.negative_label, device=self.device, rng=self.rng) # generate indices to form a batch
    
    def __len__(self) -> int:
        return len(self.batch_sampler)
    
    def _add_train_mask(self, x: Data) -> Data:
        train_mask = torch.zeros(x.y.shape[0], dtype=bool)
        train_mask[x.input_id] = True
        x.train_mask = train_mask

        return x
    
    # @TODO: train mask = consider only the central nodes?
    def __iter__(self) -> Data:
        for batch in self.batch_sampler:
            yield next(iter(NeighborLoader(self.data, num_neighbors=self.neighbourhood_sizes, input_nodes=batch, batch_size=self.batch_size, transform=self._add_train_mask, generator=self.rng))).to(self.device)
        
