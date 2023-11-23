import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

def obj_to_dict(obj) -> dict:
    return {'name':obj.__class__.__name__, 'params':{a:str(b) for a, b in obj.__dict__.items()}}

class Convolution:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        
    def update_random_params(self) -> None:
        pass

    def __repr__(self) -> str:
        d = self.to_dict()
        return f'{d["type"]}({d["name"]}, {d["params"]})'

    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels']}}


class GCNConv(Convolution):
    def init(self) -> 'GCNConv':
        self.torch_obj_instance = tg_nn.GCNConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])


class ChebConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(2, 20)
        
    def update_random_params(self) -> None:
        self.K = random.randint(2, 20)
        self.init()

    def init(self) -> 'ChebConv':
        self.torch_obj_instance = tg_nn.ChebConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                self.K)

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

    
    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'K':self.K}}


class SAGEConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.aggr = self.random_aggregation()

    def random_aggregation(self) -> 'aggr':
        default_aggr = [
            tg_nn.aggr.SumAggregation(),
            tg_nn.aggr.MeanAggregation(),
            tg_nn.aggr.MaxAggregation(),
            tg_nn.aggr.MinAggregation(),
            tg_nn.aggr.MinAggregation(),
            tg_nn.aggr.MulAggregation(),
            tg_nn.aggr.VarAggregation(semi_grad = random.choice([True, False])),
            tg_nn.aggr.StdAggregation(semi_grad = random.choice([True, False])),
            tg_nn.aggr.SoftmaxAggregation(t = random.randint(1, 10)/10, learn = random.choice([True, False])),
            tg_nn.aggr.PowerMeanAggregation(learn = True),
            tg_nn.aggr.MedianAggregation(),
            tg_nn.aggr.QuantileAggregation(random.randint(1, 100)/100, interpolation = random.choice(["lower", "higher", "midpoint", "nearest", "linear"]))
        ]

        return random.choice(default_aggr + [tg_nn.aggr.MultiAggregation([random.choice(default_aggr) for _ in range(random.randint(3, 6))])])

    def update_random_params(self) -> None:
        self.aggr = self.random_aggregation()
        self.init()


    def init(self) -> 'SAGEConv':
        self.torch_obj_instance = tg_nn.SAGEConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                aggr = self.aggr)

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

    
    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'aggr':obj_to_dict(self.aggr)}}

class GraphConv(SAGEConv):
    def init(self) -> 'GraphConv':
        self.torch_obj_instance = tg_nn.GraphConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                aggr = self.aggr)

        return self


class GatedGraphConv(SAGEConv):
    def __init__(self, genotype:'GraphGenotype') -> None:
        super().__init__(genotype)
        self.num_layers = random.randint(2, 10)

    def init(self) -> 'GatedGraphConv':
        self.torch_obj_instance = tg_nn.GatedGraphConv(
                self.genotype.network_state['out_channels'],
                self.num_layers,
                aggr = self.aggr)

        return self

    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'aggr':obj_to_dict(self.aggr), 'num_layers':self.num_layers}}


class GATConv(GCNConv):
    def init(self) -> 'GATConv':
        self.torch_obj_instance = tg_nn.GATConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self


class CuGraphGATConv(GCNConv):
    def init(self) -> 'CuGraphGATConv':
        self.torch_obj_instance = tg_nn.CuGraphGATConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self
    

class GATv2Conv(GCNConv):
    def init(self) -> 'GATv2Conv':
        self.torch_obj_instance = tg_nn.GATv2Conv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self

class TransformerConv(GCNConv):
    def init(self) -> 'TransformerConv':
        self.torch_obj_instance = tg_nn.TransformerConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self


class TAGConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(2, 7)
        
    def update_random_params(self) -> None:
        self.K = random.randint(2, 7)
        self.init()

    def init(self) -> 'TAGConv':
        self.torch_obj_instance = tg_nn.TAGConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                K = self.K)

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

    
    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'K':self.K}}

class ARMAConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.num_stacks = random.randint(2, 5)
        self.num_layers = random.randint(2, 5)
        self.dropout = random.randint(0, 5)/10
        
    def update_random_params(self) -> None:
        self.num_stacks = random.randint(2, 5)
        self.num_layers = random.randint(2, 5)
        self.dropout = random.randint(0, 5)/10
        self.init()

    def init(self) -> 'ARMAConv':
        self.torch_obj_instance = tg_nn.ARMAConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                num_stacks = self.num_stacks,
                num_layers = self.num_layers,
                dropout = self.dropout)

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

    
    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'num_stacks':self.num_stacks,
                        'num_layers':self.num_layers,
                        'dropout':self.dropout}}


class SGConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(1, 5)
        
    def update_random_params(self) -> None:
        self.K = random.randint(1, 5)
        self.init()

        
    def init(self) -> 'ARMAConv':
        self.torch_obj_instance = tg_nn.SGConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'],
                K = self.K)

        return self


    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'K':self.K}}


class APPNP(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(1, 5)
        self.alpha = random.randint(1,100)/100
        self.dropout = random.randint(0, 5)/10

    def update_random_params(self) -> None:
        self.K = random.randint(1, 5)
        self.alpha = random.randint(1,100)/100
        self.dropout = random.randint(0, 5)/10
        self.init()

    def init(self) -> 'APPNP':
        self.torch_obj_instance = tg_nn.APPNP(
                self.K, 
                self.alpha,
                dropout = self.dropout)

        return self

    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'K':self.K, 'alpha':self.alpha, 'dropout':self.dropout}}
    

class MFConv(GCNConv):
    def init(self) -> 'MFConv':
        self.torch_obj_instance = tg_nn.MFConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self