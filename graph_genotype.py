import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn
import genotype_convolutions as g_c
import genotype_normalizations as g_n
import genotype_pooling as g_p
import genotype_aug as g_a
import genotype_activations as g_ac

'''
1. convolution layers:
    for each:
        - transform (i.e torch.nn.Linear, optional)
        - activate (i.e relu, required, https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)
        - normalize (optional)
        - dropout (optional)

        If parent is the last convolutional layer, do not perform any of the steps above

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

G_LAYERS = {'convolutions':[
        (g_c.GCNConv, p(1)),
        (g_c.ChebConv, p(1)),
        (g_c.SAGEConv, p(6)),
        (g_c.GraphConv, p(6)),
        (g_c.GatedGraphConv, p(6)),
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
    ],
    'transforms':[
        (g_a.Linear, p(1)),
    ],
    'activations':[
        (g_ac.relu, p(1)),
        (g_ac.hardswish, p(1)),
        (g_ac.elu, p(1)),
        (g_ac.selu, p(1)),
        (g_ac.celu, p(1)),
        (g_ac.rrelu, p(1)),
        (g_ac.logsigmoid, p(1)),
        (g_ac.hardshrink, p(1)),
        (g_ac.softplus, p(1)),
        (g_ac.tanh, p(1)),
        (g_ac.sigmoid, p(1)),
        (g_ac.silu, p(1)),
        (g_ac.mish, p(1))
    ]
    'dropout':[
        (g_a.dropout, p(1)),
    ]
}

class GraphGenotype:
    def __init__(self, global_params:typing.Optional[dict] = {}) -> None:
        self.network_state = global_params
        
        self.genotype.network_state['x']
        self.genotype.network_state['edge_index']
        self.genotype.network_state['in_channels']
        self.genotype.network_state['out_channels']
        self.genotype.network_state['training']
        self.genotype.network_state['batch_size']
        self.genotype.network_state['num_node_features']
        self.genotype.network_state['num_classes']


if __name__ == '__main__':
    gn = GraphGenotype({'in_channels':32, 'out_channels':32, 'num_node_features':9, 'num_classes':12, 'batch_size':None})
    '''
    for [a] in d['pooling']:
        m = a(gn)
        print(m)
    '''
    p = PropCounter()
    v = [p(1), p(1), p(1)]
    print([i.P for i in v])





