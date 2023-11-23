import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn

class Normalization:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None

    def update_random_params(self) -> None:
        return False

    def to_dict(self) -> dict:
        return {'type':'normalization', 
            'name':self.__class__.__name__, 
            'params':{'in_channels':self.genotype.network_state['in_channels']}}

    def __repr__(self) -> str:
        d = self.to_dict()
        return f'{d["type"]}({d["name"]}, {d["params"]})'


class BatchNorm(Normalization):
    def init(self) -> 'BatchNorm':
        self.torch_obj_instance = tg_nn.BatchNorm(
            self.genotype.network_state['in_channels'])

        return self

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'])


class InstanceNorm(BatchNorm):
    def init(self) -> 'InstanceNorm':
        self.torch_obj_instance = tg_nn.InstanceNorm(
            self.genotype.network_state['in_channels'])

        return self


class LayerNorm(BatchNorm):
    def init(self) -> 'LayerNorm':
        self.torch_obj_instance = tg_nn.LayerNorm(
            self.genotype.network_state['in_channels'])

        return self


class GraphNorm(BatchNorm):
    def init(self) -> 'GraphNorm':
        self.torch_obj_instance = tg_nn.GraphNorm(
            self.genotype.network_state['in_channels'])

        return self


class GraphSizeNorm(BatchNorm):
    def init(self) -> ' GraphSizeNorm':
        self.torch_obj_instance = tg_nn.GraphSizeNorm()
        return self


class PairNorm(Normalization):
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.scale = self.random_scale()

    def random_scale(self) -> float:
        return random.randint(40, 250)/100

    def init(self) -> 'PairNorm':
        self.torch_obj_instance = tg_nn.PairNorm(
            scale = self.scale)

        return self
    
    def update_random_params(self) -> None:
        self.scale = self.random_scale()
        return self.init()

    def to_dict(self) -> dict:
        return {'type':'normalization', 
            'name':self.__class__.__name__, 
            'params':{'in_channels':self.genotype.network_state['in_channels'],
                'scale':self.scale}}


class MeanSubtractionNorm(BatchNorm):
    def init(self) -> 'MeanSubtractionNorm':
        self.torch_obj_instance = tg_nn.MeanSubtractionNorm(
            self.genotype.network_state['in_channels'])

        return self


class DiffGroupNorm(Normalization):
    '''https://amaarora.github.io/posts/2020-08-09-groupnorm.html'''
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None
        self.random_group_chunk()

    def random_group_chunk(self) -> int:
        self.groups = random.randint(2, 32)


    def init(self) -> 'DiffGroupNorm':
        self.torch_obj_instance = tg_nn.DiffGroupNorm(
            self.genotype.network_state['in_channels'], self.groups)

        return self
    
    def update_random_params(self) -> None:
        self.random_group_chunk
        return self.init()

    def to_dict(self) -> dict:
        return {'type':'normalization', 
            'name':self.__class__.__name__, 
            'params':{'in_channels':self.genotype.network_state['in_channels'],
                'groups':self.groups}}