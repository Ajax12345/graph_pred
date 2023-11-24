import graph_genotype, torch
import torch_geometric
import numpy as np, pickle
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
import torch.nn as nn, json
import torch.optim as optim
from sklearn.metrics import roc_auc_score

train_dataset = torch.load("datasets/train_data.pt")
valid_dataset = torch.load("datasets/valid_data.pt")
test_dataset = torch.load("datasets/test_data.pt")


batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.GG = graph_genotype.GraphGenotype.random_gnn()
        print(json.dumps(self.GG.to_dict(), indent=4))
        self.GG['hidden_channels'] = hidden_channels
        self.GG['in_channels'] = hidden_channels
        self.GG['out_channels'] = hidden_channels
        self.GG['num_node_features'] = num_node_features
        self.GG['num_classes'] = num_classes
        self.emb = AtomEncoder(hidden_channels=32)
        self.GG.init()

    def forward(self, batch):
        x , edge_index, batch_size = batch.x, batch.edge_index, batch.batch
        self.GG['x'] = x
        self.GG['edge_index'] = edge_index
        self.GG['batch_size'] = batch_size
        self.GG['training'] = self.training
        self.GG['x'] = self.emb(self.GG['x'])

        for i, layer in enumerate(self.GG.convolution_layers):
            #print('layer', i)
            layer.execute(len(self.GG.convolution_layers) - 1 == i)

        self.GG.readout_layer.execute()
        self.GG.final_dropout.execute()
        self.GG.transformation.execute()
        
        return self.GG['x']


def train(model, device, loader, optimizer, criterion):
    #model = model.to(device)
    model.train()

    for step, batch in enumerate(loader):
        #batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]).mean()
        loss.backward()
        optimizer.step()



def eval(model, device, loader, criterion):
    #model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    # For every batch in test loader
    for batch in loader:

        #batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape))
            y_pred.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    # Compute the ROC - AUC score and store as history
    rocauc_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list)/len(rocauc_list)}

def run_training(model) -> None:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Start training...")
    for epoch in range(1, 5):
        print("====epoch " + str(epoch))

        # training
        train(model, device, train_loader, optimizer, criterion)

        # evaluating
        train_acc = eval(model, device, train_loader, criterion)
        val_acc = eval(model, device, val_loader, criterion)
        print({'Train': train_acc, 'Validation': val_acc})

if __name__ == '__main__':
    model = GCN(32, 9, 12)
    run_training(model)