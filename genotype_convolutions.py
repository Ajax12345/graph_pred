import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

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
                        'aggr':self.aggr}}

    
