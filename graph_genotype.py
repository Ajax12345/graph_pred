import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

'''
1. convolution layers:
    for each:
        - activate (i.e relu, required)
        - normalize (optional)
        - dropout (optional)

2. Readout layer
    - global_mean_pool

3. Transformation
    torch.nn.Linear

'''

class GraphGenotype:
    def __init__(self, global_params:typing.Optional[dict] = {}) -> None:
        self.global_params = global_params


