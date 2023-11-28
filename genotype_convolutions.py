import typing, collections
import torch, random, re
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

def obj_to_dict(obj) -> dict:
    return {'name':obj.__class__.__name__, 'params':str(obj)}

def to_str_params(s):
        return re.sub('(?<=[a-zA-Z]\=)\w+', lambda x:f'"{x.group()}"', s)

def eval_aggr(s):
    return eval(re.sub('\w+Aggregation', lambda x:'tg_nn.aggr.'+x.group(), to_str_params(s)).replace('\n', ''))

class Convolution:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        
    def update_random_params(self) -> None:
        return False

    def __repr__(self) -> str:
        d = self.to_dict()
        return f'{d["type"]}({d["name"]}, {d["params"]})'

    def to_dict(self) -> dict:
        return {'type':'convolution', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels']}}

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        return cls(GG)

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
        if self.K < 3:
            self.K += random.randint(1, 3)
        
        else:
            self.K += random.choice([-1, 1])*random.randint(1, 3)
        
        #self.K = random.randint(2, 20)
        return True

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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.K = int(d['params']['K'])
        return gg


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
            #tg_nn.aggr.QuantileAggregation(random.randint(1, 100)/100, interpolation = random.choice(["lower", "higher", "midpoint", "nearest", "linear"]))
        ]

        return random.choice(default_aggr + [tg_nn.aggr.MultiAggregation([random.choice(default_aggr) for _ in range(random.randint(3, 6))])])

    def update_random_params(self) -> None:
        self.aggr = self.random_aggregation()
        return True

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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.aggr = eval_aggr(d['params']['aggr']['params'])
        return gg


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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.aggr = eval_aggr(d['params']['aggr']['params'])
        gg.num_layers = int(d['params']['num_layers'])
        return gg

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
        if self.K < 3:
            self.K += random.randint(1, 3)
        
        else:
            self.K += random.choice([-1, 1])*random.randint(1, 3)

        #self.K = random.randint(2, 7)
        return True

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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.K = int(d['params']['K'])
        return gg

class ARMAConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.num_stacks = random.randint(2, 5)
        self.num_layers = random.randint(2, 5)
        self.dropout = random.randint(5, 50)/100
        
    def update_random_params(self) -> None:
        if self.num_stacks < 3:
            self.num_stacks += random.randint(1, 2)
        else:
            self.num_stacks += random.choice([-1, 1])*random.randint(1, 2)
        
        if self.num_layers < 3:
            self.num_layers += random.randint(1, 2)
        else:
            self.num_layers += random.choice([-1, 1])*random.randint(1, 2)
        
        if self.dropout < 0.1:
            self.dropout += random.randint(5, 50)/100

        else:
            self.dropout += random.choice([-1, 1])*random.randint(5, int(self.dropout*100))/100
        
        #self.num_stacks = random.randint(2, 5)
        #self.num_layers = random.randint(2, 5)
        #self.dropout = random.randint(0, 5)/10
        return True

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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.num_stacks = int(d['params']['num_stacks'])
        gg.num_layers = int(d['params']['num_layers'])
        gg.dropout = float(d['params']['dropout'])
        return gg

class SGConv(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(1, 5)
        
    def update_random_params(self) -> None:
        if self.K < 3:
            self.K += random.randint(1, 3)
        
        else:
            self.K += random.choice([-1, 1])*random.randint(1, 3)
        #self.K = random.randint(1, 5)
        return True
        
    def init(self) -> 'ARMAConv':
        self.torch_obj_instance = tg_nn.SGConv(
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

    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.K = int(d['params']['K'])
        return gg

class APPNP(Convolution):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.K = random.randint(1, 5)
        self.alpha = random.randint(1,100)/100
        self.dropout = random.randint(5, 50)/100

    def update_random_params(self) -> None:
        if self.K < 3:
            self.K += random.randint(1, 3)
        
        else:
            self.K += random.choice([-1, 1])*random.randint(1, 3)


        if self.alpha < 0.1:
            self.alpha += random.randint(5, 50)/100

        else:
            self.alpha += random.choice([-1, 1])*random.randint(5, int(self.alpha*100))/100
        

        if self.dropout < 0.1:
            self.dropout += random.randint(5, 50)/100

        else:
            self.dropout += random.choice([-1, 1])*random.randint(5, int(self.dropout*100))/100
        
        #self.K = random.randint(1, 5)
        self.alpha = random.randint(1,100)/100
        #self.dropout = random.randint(0, 5)/10
        return True

    def init(self) -> 'APPNP':
        self.torch_obj_instance = tg_nn.APPNP(
                self.K, 
                self.alpha,
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
                        'K':self.K, 'alpha':self.alpha, 'dropout':self.dropout}}
    
    @classmethod
    def from_dict(cls, GG, d:dict) -> 'Convolution':
        gg = cls(GG)
        gg.K = int(d['params']['K'])
        gg.alpha = float(d['params']['alpha'])
        gg.dropout = float(d['params']['dropout'])
        return gg

class MFConv(GCNConv):
    def init(self) -> 'MFConv':
        self.torch_obj_instance = tg_nn.MFConv(
                self.genotype.network_state['in_channels'], 
                self.genotype.network_state['out_channels'])

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

if __name__ == '__main__':
    s = """
MultiAggregation([
    MulAggregation(),
    MaxAggregation(),
    MeanAggregation(),
    MinAggregation(),
    StdAggregation(),
    MaxAggregation(),
    ], mode=cat)
    """
    def to_str_params(s):
        return re.sub('(?<=[a-zA-Z]\=)\w+', lambda x:f'"{x.group()}"', s)

    def eval_aggr(s):
        return eval(re.sub('\w+Aggregation', lambda x:'tg_nn.aggr.'+x.group(), to_str_params(s)).replace('\n', ''))

    print(eval_aggr(s).__class__)

    
