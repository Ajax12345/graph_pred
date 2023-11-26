import typing, collections
import torch, random, json
import torch_geometric
import torch.nn as nn
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

3. Final dropout

4. Transformation
    torch.nn.Linear

'''

'''
dependencies:
    -  pynndescent
    conda install -c nvidia pylibcugraphops (needed for CuGraphGATConv)
'''

class PropCounter:
    def __init__(self) -> None:
        self.vals = []
        self.s = None
    
    def __call__(self, weight:int) -> 'PropCounter':
        self.vals.append(weight)

        class Prop:
            def __init__(self, w:int, _self) -> None:
                self.w = w
                self._c = _self

            @property
            def P(self) -> int:
                return self.w/sum(self._c.vals)

        return Prop(weight, self)

p = PropCounter()

G_LAYERS = {'convolutions':[
        (g_c.GCNConv, p(6)),
        (g_c.ChebConv, p(1)),
        (g_c.SAGEConv, p(6)),
        (g_c.GraphConv, p(6)),
        (g_c.GatedGraphConv, p(5)),
        (g_c.GATConv, p(1)),
        #(g_c.CuGraphGATConv, p(1)), unable to install module
        (g_c.GATv2Conv, p(1)),
        (g_c.TransformerConv, p(1)),
        (g_c.TAGConv, p(1)),
        (g_c.ARMAConv, p(1)),
        (g_c.SGConv, p(1)),
        (g_c.APPNP, p(1)),
        (g_c.MFConv, p(1)),
    ],
    'normalizations':[
        (g_n.BatchNorm, p(4)),
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
        (g_p.global_mean_pool, p(4)),
        (g_p.global_max_pool, p(3)),
    ],
    'transforms':[
        (g_a.Linear, p(1)),
    ],
    'activations':[
        (g_ac.relu, p(10)),
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
    ],
    'dropout':[
        (g_a.dropout, p(1)),
    ]
}

'''
self.genotype.network_state['x']
self.genotype.network_state['edge_index']
self.genotype.network_state['in_channels']
self.genotype.network_state['out_channels']
self.genotype.network_state['training']
self.genotype.network_state['batch_size']
self.genotype.network_state['num_node_features']
self.genotype.network_state['num_classes']
'''

def random_choice(options:typing.List[typing.Tuple['obj', PropCounter]]) -> 'obj':
    a, p = zip(*options)
    return random.choices([*a], weights = [i.P for i in p], k = 1)[0]


class g_layer_wrapper:
    @property
    def r_convolution(self) -> typing.Any:
        return random_choice(G_LAYERS['convolutions'])

    @property
    def r_normalization(self) -> typing.Any:
        return random_choice(G_LAYERS['normalizations'])

    @property
    def r_pool(self) -> typing.Any:
        return random_choice(G_LAYERS['pooling'])

    @property
    def r_transform(self) -> typing.Any:
        return random_choice(G_LAYERS['transforms'])

    @property
    def r_activation(self) -> typing.Any:
        return random_choice(G_LAYERS['activations'])

    @property
    def r_dropout(self) -> typing.Any:
        return random_choice(G_LAYERS['dropout'])
    

G_L = g_layer_wrapper()

class ConvolutionLayer:
    def __init__(self, GG, convolution, transform = None, activate = None, 
                        normalize = None, dropout = None) -> None:
        self.convolution = convolution
        self.transform = transform
        self.activate = activate
        self.normalize = normalize
        self.dropout = dropout
        self.GG = GG
        if self.activate is None:
            raise Exception('activation must be specified')

    def init(self) -> None:
        for a, b in self.__dict__.items():
            if a != 'GG' and hasattr(b, 'init'):
                self.__dict__[a] = b.init()

        return self

    def update_random_params(self, p = 0.2) -> None:
        l = [self.convolution.update_random_params()]
        
        if random.random() <= p:
            if self.transform is None:
                self.transform = G_L.r_transform(self.GG)

            else:
                self.transform = None
            
            l.append(True)

        if random.random() <= p:
            self.activate = G_L.r_activation(self.GG)
            l.append(True)

        if random.random() <= p:
            if self.normalize is None:
                self.normalize = G_L.r_normalization(self.GG)
            
            else:
                self.normalize = None
            
            l.append(True)

        if random.random() <= p:
            if self.dropout is None:
                self.dropout = G_L.r_dropout(self.GG)

            else:
                self.dropout = None
            
            l.append(True)

        return any(l)

    def execute(self, last = False) -> None:
        self.convolution.execute()
        if hasattr(self.transform, 'execute') and not last:
            self.transform.execute()

        if hasattr(self.activate, 'execute') and not last:
            self.activate.execute()

        if hasattr(self.normalize, 'execute') and not last:
            self.normalize.execute()

        if hasattr(self.dropout, 'execute') and not last:
            self.dropout.execute()

        return self

    @classmethod
    def random_layer(cls, GG, transform_prob = 0.4, normalize_prob = 0.5, dropout_prob = 0.5) -> 'ConvolutionLayer':
        convolution = G_L.r_convolution(GG)
        transform = None
        activate = G_L.r_activation(GG)
        normalize = None
        dropout = None

        if random.random() <= transform_prob:
            transform = G_L.r_transform(GG)

        if random.random() <= normalize_prob:
            normalize = G_L.r_normalization(GG)
        
        if random.random() <= dropout_prob:
            dropout = G_L.r_dropout(GG)

        return cls(
            GG,
            convolution,
            transform = transform,
            activate = activate,
            normalize = normalize,
            dropout = dropout
        )

    def to_dict(self) -> dict:
        return {'type':self.__class__.__name__,
                **{a:b if b is None else b.to_dict() for a, b in self.__dict__.items() if a != 'GG'}}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({", ".join(a+" = "+b.__class__.__name__ for a, b in self.__dict__.items() if a != "GG")})'
    
class W:
    def __init__(self, gp):
        self.gp = gp

    def __getitem__(self, n):
        return self.gp.get(n)

    def __setitem__(self, n, a):
        self.gp[n] = a


class GraphGenotype:
    def __init__(self, global_params:typing.Optional[dict] = {}) -> None:
        self.network_state = W(global_params)
        self.convolution_layers = None
        self.readout_layer = None
        self.final_dropout = None
        self.transformation = None

    def add_layer(self) -> bool:
        assert self.convolution_layers is not None
        self.convolution_layers.append(ConvolutionLayer.random_layer(self))
        return True

    def remove_layer(self) -> bool:
        assert self.convolution_layers is not None
        if len(self.convolution_layers) == 1:
            return False

        I = random.randint(0, len(self.convolution_layers) - 1)
        self.convolution_layers = self.convolution_layers[:I]+self.convolution_layers[I+1:]
        return True

    def swap_layers(self) -> bool:
        assert self.convolution_layers is not None
        if len(self.convolution_layers) == 1:
            return False

        I = random.randint(0, len(self.convolution_layers) - 2)
        I1 = random.randint(I+1, len(self.convolution_layers) - 1)
        l = self.convolution_layers[I1]
        self.convolution_layers[I1] = self.convolution_layers[I]
        self.convolution_layers[I] = l
        return True

    def update_layers(self) -> bool:
        assert self.convolution_layers is not None
        layers = [*range(len(self.convolution_layers))]
        while layers:
            I = random.choice(layers)
            if self.convolution_layers[I].update_random_params():
                print('update mutation ind', I)
                return True

            layers.remove(I)
        

        I = random.randint(0, len(self.convolution_layers) - 1)
        print('layer update', I)
        self.convolution_layers[I] = ConvolutionLayer.random_layer(self)
        
        return True


    def mutate(self) -> None:
        #options = [('add_layer', 0.7), ('remove_layer', 0.05), ('swap_layers', 0.05), ('update_layers', 0.2)]
        options = [('add_layer', 0.7), ('remove_layer', 0.1), ('swap_layers', 0.15), ('update_layers', 0.05)]
        #options = [('add_layer', 0.7), ('remove_layer', 0.1), ('swap_layers', 0.2)]
        methods, probs = zip(*options)
        getattr(self, random.choices(methods, weights = probs, k=1)[0])()
               

    @classmethod
    def all_modules(cls, obj) -> typing.Iterator:
        if hasattr(obj, 'torch_obj_instance'):
            if isinstance(obj.torch_obj_instance, nn.Module):
                yield obj.torch_obj_instance

            return

        if isinstance(obj, list):
            for i in obj:
                yield from cls.all_modules(i)

            return
        
        for a, b in getattr(obj, '__dict__', {}).items():
            if a != 'GG':
                yield from cls.all_modules(b)

    def __getitem__(self, n:str) -> typing.Any:
        return self.network_state[n]

    def __setitem__(self, n:str, a:typing.Any) -> None:
        self.network_state[n] = a
        
    def init(self) -> None:
        for layer in self.convolution_layers:
            layer.init()

        self.readout_layer.init()
        self.final_dropout.init()
        self.transformation.init()

    @classmethod
    def random_gnn(cls, layers = (2, 5)) -> 'GraphGenotype':
        GG = cls()
        GG.convolution_layers = [ConvolutionLayer.random_layer(GG) for _ in range(random.randint(*layers))]
        GG.readout_layer = G_L.r_pool(GG)
        GG.final_dropout = G_L.r_dropout(GG)
        GG.transformation = g_a.LinearFinal(GG)
        return GG

    def purge(self) -> None:
        self['hidden_channels'] = None
        self['in_channels'] = None
        self['out_channels'] = None
        self['num_node_features'] = None
        self['num_classes'] = None
        self['x'] = None
        self['edge_index'] = None
        self['batch_size'] = None
        self['training'] = None

    def to_dict(self) -> dict:
        return {
            'layers':[i.to_dict() for i in self.convolution_layers],
            'readout_layer':self.readout_layer.to_dict(),
            'final_dropout':self.final_dropout.to_dict(),
            'transformation':self.transformation.to_dict()
        }

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(layer_num = {len(self.layers)})'


if __name__ == '__main__':
    '''
    for [a] in d['pooling']:
        m = a(gn)
        print(m)
    '''
    gn = GraphGenotype.random_gnn()
    print(json.dumps(gn.to_dict(), indent=4))
    print('-'*50)
    gn.mutate()
    print(json.dumps(gn.to_dict(), indent=4))
    





