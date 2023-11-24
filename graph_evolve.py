import graph_genotype, torch
import torch_geometric
import numpy as np, pickle
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score