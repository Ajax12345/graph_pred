import typing, json, os, graph_genotype as g_g
import matplotlib.pyplot as plt, datetime, re

def best_graph() -> 'Genotype':
    results = []
    for i in os.listdir('generated_graphs'):
        if i.startswith('results'):
            print(i)
            with open(os.path.join('generated_graphs', i, 'results.json')) as f:
                graph = json.load(f)
                results.extend(graph)


    #print(results)
    score, graph = max(results, key=lambda x:float(x[0]))
    print('best graph score', score)
    #print(json.dumps(graph, indent=4))
    #print('-'*40)
    return g_g.GraphGenotype.from_dict(graph)
    print(json.dumps(GG.to_dict(), indent=4))

def plot_accuracies() -> None:
    results = []
    results1 = []
    for i in os.listdir('generated_graphs'):
        if i.startswith('results'):
            with open(os.path.join('generated_graphs', i, 'results.json')) as f:
                try:
                    graph = json.load(f)
                    if re.findall('results\d{4,}', i):
                        results1.append([datetime.datetime.fromtimestamp(int(re.findall('\d+', i)[0])), max(graph, key=lambda x:x[0])[0]])
                    else:
                        results.append(max(graph, key=lambda x:x[0])[0])
                except:
                    pass

    k = results + [b for _, b in sorted(results1, key=lambda x:x[0])]
    plt.plot([*range(1, len(k)+1)], k)
    plt.show()
        

if __name__ == '__main__':
    plot_accuracies()