import typing, json, os, graph_genotype as g_g


def best_graph() -> None:
    results = []
    for i in os.listdir('generated_graphs'):
        if i.startswith('results'):
            with open(os.path.join('generated_graphs', i, 'results.json')) as f:
                try:
                    graph = json.load(f)
                    results.extend(graph)
                except:
                    print(i)

            
    print(json.dumps(max(results, key=lambda x:x[0])[1], indent=4))


if __name__ == '__main__':
    best_graph()