import networkx as nx
import json


def load_graph(path: str):
    with open(path, 'r') as f:
        json_data = json.load(f)
    
    graph = json_data['graph']
    
    G = nx.DiGraph()
    for i in range(graph['nodes']):
        G.add_node(i, x=graph[str(i)]['x'], y=graph[str(i)]['y'])
        for j, w in graph[str(i)]['adj']:
            G.add_edge(i, j, weight=w)

    return G


def load_constraints(path: str):
    with open(path, 'r') as f:
        json_data = json.load(f)

    return json_data['max_cost']

def load_generator_data(path: str):
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data['graph']['generator']['min_w'], json_data['graph']['generator']['max_w']

def load_metro_params(path: str):
    with open(path, 'r') as f:
        json_data = json.load(f)

    return json_data['metro']
