# Atom encoder

class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(torch.nn.Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


# A simple graph neural network model

from torch_geometric.nn import GCNConv, TransformerConv
import torch_geometric.nn as tg_nn
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F
from torch.nn import Linear
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = tg_nn.GatedGraphConv(hidden_channels, 4, aggr = tg_nn.aggr.MinAggregation())


        self.conv2 = tg_nn.GraphConv(hidden_channels, hidden_channels, aggr = tg_nn.aggr.SumAggregation())
        self.bn2 = tg_nn.BatchNorm(hidden_channels)

        self.conv3 = tg_nn.GatedGraphConv(hidden_channels, 5, aggr = tg_nn.aggr.MulAggregation())
        self.lin3 = Linear(hidden_channels, hidden_channels)


        self.conv4 = tg_nn.TAGConv(hidden_channels, hidden_channels, K = 7)
        self.lin4 = Linear(hidden_channels, hidden_channels)

        self.conv5 = tg_nn.SAGEConv(hidden_channels, hidden_channels, aggr = tg_nn.aggr.VarAggregation())
        self.lin5 = Linear(hidden_channels, hidden_channels)
        self.bn5 = tg_nn.InstanceNorm(hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, batch):
        x , edge_index, batch_size = batch.x, batch.edge_index, batch.batch
        x = self.emb(x)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.bn2(x)

        x = self.conv3(x, edge_index)
        x = self.lin3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = self.lin4(x)
        x = F.softplus(x)

        x = self.conv5(x, edge_index)
        x = self.lin5(x)
        x = F.softplus(x)
        x = self.bn5(x)
        x = F.dropout(x, p=0.5, training=self.training)


        # 2. Readout layer
        x = tg_nn.global_max_pool(x, batch_size)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GCN_OLD(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.emb = AtomEncoder(hidden_channels=32)
        self.conv1 = GCNConv(hidden_channels,hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    def forward(self, batch):
        x , edge_index, batch_size = batch.x, batch.edge_index, batch.batch
        x = self.emb(x)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = gap(x, batch_size)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x