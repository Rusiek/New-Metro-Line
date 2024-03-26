import networkx as nx
import numpy as np
import itertools
import json


def generate_consecutive(path: str, size: int = 4):
    json_data = {}
    graph = {}
    for i in range(size):
        graph['nodes'] = size
        graph['edges'] = size - 1
        graph[i] = {}
        graph[i]['x'] = i
        graph[i]['y'] = i % 2
        graph[i]['adj'] = []
        if i < size - 1:
            graph[i]['adj'].append((i + 1, np.random.randint(5, 10)))
        if i > 0:
            graph[i]['adj'].append((i - 1, np.random.randint(5, 10)))
    json_data['graph'] = graph
    metro = {}
    metro['time/km'] = 5
    metro['cost/km'] = 10
    metro['cost/station'] = 10
    json_data['metro'] = metro
    json_data['max_cost'] = 30
    with open(f'{path}.json', 'w') as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    np.random.seed(0)
    for i in range(3, 7):
        generate_consecutive(f'benchmark/small/consecutive_{i}', i)
