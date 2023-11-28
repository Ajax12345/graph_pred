import graph_genotype, torch
import torch_geometric, traceback
import numpy as np, pickle, copy
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader
import torch.nn as nn, json, random
import torch.optim as optim, os, time
from sklearn.metrics import roc_auc_score
import genotype_retrieve, typing


batch_size=32

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
    def __init__(self, GG, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.GG = GG
        #print(json.dumps(self.GG.to_dict(), indent=4))
        self.GG['hidden_channels'] = hidden_channels
        self.GG['in_channels'] = hidden_channels
        self.GG['out_channels'] = hidden_channels
        self.GG['num_node_features'] = num_node_features
        self.GG['num_classes'] = num_classes
        self.layers = nn.ModuleList()
        self.emb = AtomEncoder(hidden_channels=32)
        self.GG.init()
        for i in graph_genotype.GraphGenotype.all_modules(self.GG):
            #print('module here', i)
            self.layers.append(i)

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
            #layer.execute()

        self.GG.readout_layer.execute()
        self.GG.final_dropout.execute()
        self.GG.transformation.execute()
        
        return self.GG['x']


def train(model, device, loader, optimizer, criterion):
    model = model.to(device)
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        optimizer.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]).mean()
        loss.backward()
        optimizer.step()



def eval(model, device, loader, criterion):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    # For every batch in test loader
    for batch in loader:

        batch = batch.to(device)
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

def run_training(train_loader, val_loader, model, lr, train_evolutions) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('mps')
    #print("Start training...")
    results = []
    for epoch in range(1, train_evolutions+1):
        #print("====epoch " + str(epoch))

        # training
        train(model, device, train_loader, optimizer, criterion)

        # evaluating
        train_acc = eval(model, device, train_loader, criterion)
        val_acc = eval(model, device, val_loader, criterion)
        results.append({'Train': train_acc, 'Validation': val_acc})

    return results

def run_evolutionary_process(train_loader, val_loader, pop_size = 10, iterations = 10, train_evolutions = 4, prob_mutations = 0.4, parent_GG:typing.Union[None, 'Genotype'] = None) -> None:
    #model = GCN(32, 9, 12)
    #graph_genotype.GraphGenotype.random_gnn()
    os.mkdir(root_path:=f'results{str(time.time()).split(".")[0]}')
    with open(os.path.join(root_path, 'results.json'), 'a') as f:
        pass

    all_results, best_results = [], []
    if parent_GG is None:
        population = [graph_genotype.GraphGenotype.random_gnn() for _ in range(pop_size)]
    
    else:
        population = [copy.deepcopy(parent_GG)]
        for _ in range(pop_size - 1):
            gg = copy.deepcopy(parent_GG)
            gg.mutate()
            population.append(gg)

    for _ in range(iterations):
        print('iteration', _ + 1)
        n_p = []
        error_count, pop_count = 0, 0
        for gg in population:
            print(gg)
            model = GCN(gg, 32, 9, 12)
            pop_count += 1
            try:
                training_results = run_training(train_loader, val_loader, model, gg.lr, train_evolutions)
                n_p.append([max(training_results, key=lambda x:x['Validation']['rocauc'])['Validation']['rocauc'], gg])
            except:
                #print(traceback.format_exc())
                error_count += 1

            print('error percentage', error_count/pop_count)

        if not n_p:
            population = [random.choice(best_results) for _ in range(pop_size)]
            continue

        score, m_gg = max(n_p, key=lambda x:x[0])
        model = GCN(m_gg, 32, 9, 12)
        try:
            training_results = run_training(train_loader, val_loader, model, m_gg.lr, 4)
            score = max(training_results, key=lambda x:x['Validation']['rocauc'])['Validation']['rocauc']
            print('best score', score)
        except:
            #print(traceback.format_exc())
            if best_results:
                population = []
                for _ in range(pop_size):
                    gg = random.choice(best_results)
                    gg.purge()
                    population.append(copy.deepcopy(gg))

            else:
                population = [graph_genotype.GraphGenotype.random_gnn() for _ in range(pop_size)]
            
            continue
            print('best score (one epoch)', score)

        all_results.append([score, m_gg.to_dict()])
        best_results.append(m_gg)
        '''
        scores, a_gg = zip(*n_p)
        s_scores = sum(scores)
        population = []
    
        for g in random.choices(a_gg, weights = [i/s_scores for i in scores], k = pop_size):
            g.mutate()
            
            population.append(g)
        '''
        population = []
        m_gg.purge()
        for _ in range(pop_size-5):
            p_gg = copy.deepcopy(m_gg)
            p_gg.mutate()
            population.append(p_gg)
        
        for _ in range(5):
            _, p_gg = random.choice(n_p)
            p_gg.purge()
            np_gg = copy.deepcopy(p_gg)
            np_gg.mutate()
            population.append(np_gg)
        
        #population.extend([random.choice(n_p)[-1] for _ in range(5)])

        with open(os.path.join(root_path, 'results.json'), 'w') as f:
            json.dump(all_results, f)

    print('final result!')
    print(max(all_results, key=lambda x:x[0]))
    with open(os.path.join(root_path, 'results.json'), 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    '''
    GG = graph_genotype.GraphGenotype.random_gnn()
    model = GCN(GG, 32, 9, 12)
    print(model)
    run_training(model)
    #print(model)
    '''
    train_dataset = torch.load("/Users/jamespetullo/graph_pred_datasets/datasets/train_data.pt")
    valid_dataset = torch.load("/Users/jamespetullo/graph_pred_datasets/datasets/valid_data.pt")
    test_dataset = torch.load("/Users/jamespetullo/graph_pred_datasets/datasets/test_data.pt")

    
    batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #https://packaging.python.org/en/latest/tutorials/packaging-projects/
    best_graph = genotype_retrieve.best_graph()
    run_evolutionary_process(train_loader, val_loader, pop_size = 30, iterations = 20, train_evolutions = 3, parent_GG = best_graph) 
    