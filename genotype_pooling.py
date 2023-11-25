import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

class Pooling:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None

    def update_random_params(self) -> None:
        return False

    def to_dict(self) -> dict:
        return {'type':'pooling', 
            'name':self.__class__.__name__, 
            'params':{'in_channels':self.genotype.network_state['in_channels'], 'batch_size':self.genotype.network_state['batch_size']}}

    def __repr__(self) -> str:
        d = self.to_dict()
        return f'{d["type"]}({d["name"]}, {d["params"]})'

class global_add_pool(Pooling):
    def init(self) -> None:
        self.torch_obj_instance = tg_nn.global_add_pool
        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'],
            self.genotype.network_state['batch_size'])


class global_mean_pool(global_add_pool):
    def init(self) -> None:
        self.torch_obj_instance = tg_nn.global_mean_pool
        return self

class global_max_pool(global_add_pool):
    def init(self) -> None:
        self.torch_obj_instance = tg_nn.global_max_pool
        return self

class TopKPooling(Pooling):
    '''
    Warning: resizing required!
    '''
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.ratio = self.random_ratio()

    def random_ratio(self) -> float:
        return random.randint(20, 100)/100

    def update_random_params(self) -> None:
        self.ratio = self.random_ratio()
        return True

    def init(self) -> 'TopKPooling':
        self.torch_obj_instance = tg_nn.TopKPooling(
                self.genotype.network_state['in_channels'], 
                ratio = self.ratio)

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'], 
            self.genotype.network_state['edge_index'])

    
    def to_dict(self) -> dict:
        return {'type':'pooling', 
                'name':self.__class__.__name__, 
                'params':{'in_channels':self.genotype.network_state['in_channels'], 
                        'out_channels':self.genotype.network_state['out_channels'], 
                        'edge_index':self.genotype.network_state['edge_index'],
                        'ratio':self.ratio}}




        