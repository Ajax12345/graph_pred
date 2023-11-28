import typing, json, os, graph_genotype as g_g


def best_graph() -> 'Genotype':
    results = []
    for i in os.listdir('generated_graphs'):
        if i.startswith('results'):
            with open(os.path.join('generated_graphs', i, 'results.json')) as f:
                try:
                    graph = json.load(f)
                    results.extend(graph)
                except:
                    pass

            
    graph = max(results, key=lambda x:x[0])[1]
    #print(json.dumps(graph, indent=4))
    #print('-'*40)
    return g_g.GraphGenotype.from_dict(graph)
    print(json.dumps(GG.to_dict(), indent=4))

if __name__ == '__main__':
    best_graph()