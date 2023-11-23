import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn
import genotype_convolutions as g_c

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




class GraphGenotype:
    def __init__(self, global_params:typing.Optional[dict] = {}) -> None:
        self.network_state = global_params



if __name__ == '__main__':
    gn = GraphGenotype({'in_channels':32, 'out_channels':32, 'num_node_features':9, 'num_classes':12})
    d = {'convolutions':[
            (g_c.GCNConv,),
            (g_c.ChebConv,),
            (g_c.SAGEConv,),
            (g_c.GraphConv,),
            (g_c.GatedGraphConv,),
            (g_c.GATConv,),
            (g_c.CuGraphGATConv,),
            (g_c.GATv2Conv,),
            (g_c.TransformerConv,),
            (g_c.TAGConv,),
            (g_c.ARMAConv,),
            (g_c.SGConv,),
            (g_c.APPNP,),
            (g_c.MFConv,),

        ]
    }
    for [a] in d['convolutions']:
        m = a(gn)
        print(m)





