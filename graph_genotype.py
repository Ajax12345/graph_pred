import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn
import genotype_convolutions as g_c
import genotype_normalizations as g_n
import genotype_pooling as g_p

'''
1. convolution layers:
    for each:
        - activate (i.e relu, required, https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)
        - normalize (optional)
        - dropout (optional)

2. Readout layer
    - global_mean_pool

3. Transformation
    torch.nn.Linear

'''

'''
dependencies:
    -  pynndescent
'''

class PropCounter:
    def __init__(self) -> None:
        self.vals = collections.deque()
        self.s = None
    
    def __call__(self, weight:int) -> 'PropCounter':
        self.vals.append(weight)
        return self

    @property
    def P(self) -> int:
        if self.s is None:
            self.s = sum(self.vals)

        return self.vals.popleft()/self.s

class GraphGenotype:
    def __init__(self, global_params:typing.Optional[dict] = {}) -> None:
        self.network_state = global_params


if __name__ == '__main__':
    gn = GraphGenotype({'in_channels':32, 'out_channels':32, 'num_node_features':9, 'num_classes':12, 'batch_size':None})
    d = {'convolutions':[
            (g_c.GCNConv, p(1)),
            (g_c.ChebConv, p(1)),
            (g_c.SAGEConv, p(1)),
            (g_c.GraphConv, p(1)),
            (g_c.GatedGraphConv, p(1)),
            (g_c.GATConv, p(1)),
            (g_c.CuGraphGATConv, p(1)),
            (g_c.GATv2Conv, p(1)),
            (g_c.TransformerConv, p(1)),
            (g_c.TAGConv, p(1)),
            (g_c.ARMAConv, p(1)),
            (g_c.SGConv, p(1)),
            (g_c.APPNP, p(1)),
            (g_c.MFConv, p(1)),
        ],
        'normalizations':[
            (g_n.BatchNorm, p(1)),
            (g_n.InstanceNorm, p(1)),
            (g_n.LayerNorm, p(1)),
            (g_n.GraphNorm, p(1)),
            (g_n.GraphSizeNorm, p(1)),
            (g_n.PairNorm, p(1)),
            (g_n.MeanSubtractionNorm, p(1)),
            (g_n.DiffGroupNorm, p(1)),
        ],
        'pooling':[
            (g_p.global_add_pool, p(1)),
            (g_p.global_mean_pool, p(1)),
            (g_p.global_max_pool, p(1)),
        ]
    }
    '''
    for [a] in d['pooling']:
        m = a(gn)
        print(m)
    '''
    p = PropCounter()
    v = [p(1), p(1), p(1)]
    print([i.P for i in v])





