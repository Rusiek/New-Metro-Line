import numpy as np
import networkx as nx


def dist(self, A, B):
    metro_dist_x = self.G.nodes[A]['x'] - self.G.nodes[B]['x']
    metro_dist_y = self.G.nodes[A]['y'] - self.G.nodes[B]['y']
    return np.sqrt(metro_dist_x ** 2 + metro_dist_y ** 2)


def get_score(self, solution):
    init_score = 0
    G = self.G.copy()

    for node_A in self.G.nodes:
        for node_B in self.G.nodes:
            init_score += nx.shortest_path_length(self.G, node_A, node_B, weight='weight')