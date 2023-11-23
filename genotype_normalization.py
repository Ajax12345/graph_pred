import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

class Normalization:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None


class BatchNorm(Normalization):
    def init(self) -> 'BatchNorm':
        self.torch_obj_instance = tg_nn.BatchNorm(s
            elf.genotype.network_state['in_channels'])

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'])