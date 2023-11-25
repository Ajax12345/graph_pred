import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn
import torch.nn as nn
import torch.nn.functional as F

class Linear:
    '''transforms'''
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None

    def update_random_params(self) -> bool:
        return False

    def init(self) -> 'nn.Linear':
        self.torch_obj_instance = nn.Linear(
            self.genotype.network_state['in_channels'],
            self.genotype.network_state['out_channels'])

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'])
    

    def to_dict(self) -> dict:
        return {'type':'transform', 
            'name':self.__class__.__name__, 
            'params':{'in_channels':self.genotype.network_state['in_channels'], 'out_channels':self.genotype.network_state['out_channels']}}


class LinearFinal(Linear):
    def init(self) -> 'nn.Linear':
        self.torch_obj_instance = nn.Linear(
            self.genotype.network_state['in_channels'],
            self.genotype.network_state['num_classes'])

        return self

class dropout:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.p = 0.5

    def update_random_params(self) -> bool:
        self.p = random.randint(35, 85)/100

        return True

    def init(self) -> 'nn.Linear':
        self.torch_obj_instance = F.dropout
        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'],
            p = self.p,
            training = self.genotype.network_state['training'])
    

    def to_dict(self) -> dict:
        return {'type':'dropout', 
            'name':self.__class__.__name__, 
            'params':{'p':self.p}}
