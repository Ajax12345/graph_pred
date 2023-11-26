import typing, collections
import torch, random
import torch_geometric
import numpy as np, torch_geometric.nn as tg_nn
import torch.nn as nn
import torch.nn.functional as F

class Activation:
    def __init__(self, genotype:'GraphGenotype') -> None:
        self.genotype = genotype
        self.torch_obj_instance = None

    def update_random_params(self) -> None:
        return False

    def to_dict(self) -> dict:
        return {'type':'activation', 
            'name':self.__class__.__name__, 
            'params':{}}

    def execute(self) -> None:
        self.genotype.network_state['x'] = self.torch_obj_instance(
            self.genotype.network_state['x'])

    def __repr__(self) -> str:
        d = self.to_dict()
        return f'{d["type"]}({d["name"]}, {d["params"]})'


class relu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.relu
        return self


class hardswish(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.hardswish
        return self

class elu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.elu
        return self


class selu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.selu
        return self


class celu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.celu
        return self


class rrelu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.rrelu
        return self


class logsigmoid(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.logsigmoid
        return self


class hardshrink(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.hardshrink
        return self


class softplus(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.softplus
        return self


class tanh(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.tanh
        return self


class sigmoid(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.sigmoid
        return self


class silu(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.silu
        return self


class mish(Activation):
    def init(self) -> None:
        self.torch_obj_instance = F.mish
        return self