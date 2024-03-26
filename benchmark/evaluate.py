import networkx as nx
import numpy as np


class Evaluate:
    def __init__(self, G, metro, max_cost):
        self.G = G
        self.metro = metro
        self.max_cost = max_cost

    def dist(self, A, B):
        metro_dist_x = self.G.nodes[A]['x'] - self.G.nodes[B]['x']
        metro_dist_y = self.G.nodes[A]['y'] - self.G.nodes[B]['y']
        return np.sqrt(metro_dist_x ** 2 + metro_dist_y ** 2)

    def evaluate(self, solution):
        init_score = 0
        for node_A in self.G.nodes:
            for node_B in self.G.nodes:
                init_score += nx.shortest_path_length(self.G, node_A, node_B, weight='weight')
        
        G_metro = self.G.copy()
        total_cost = 0
        for i in range(len(solution) - 1):
            distance = self.dist(solution[i], solution[i + 1])
            total_cost += distance * self.metro['cost/km']
            new_weigth = distance * self.metro['time/km']

            G_metro.add_edge(solution[i], solution[i + 1], weight=new_weigth)
            G_metro.add_edge(solution[i + 1], solution[i], weight=new_weigth)
        
        total_cost += len(solution) * self.metro['cost/station']
        
        final_score = 0
        for node_A in G_metro.nodes:
            for node_B in G_metro.nodes:
                final_score += nx.shortest_path_length(G_metro, node_A, node_B, weight='weight')

        output = {
            'valid': total_cost <= self.max_cost,
            'init_score': init_score,
            'final_score': final_score,
            'total_cost': total_cost
        }

        return output


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_node(0, x=0, y=0)
    G.add_node(1, x=1, y=1)
    G.add_node(2, x=2, y=0)
    G.add_node(3, x=3, y=1)
    G.add_edge(0, 1, weight=5)
    G.add_edge(1, 0, weight=7)
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 1, weight=7)
    G.add_edge(2, 3, weight=5)
    G.add_edge(3, 2, weight=7)
    metro = {'time/km': 2, 'cost/km': 10, 'cost/station': 10}
    solution = [0, 2]
    max_price = 100
    e = Evaluate(G, metro, max_price)
    print(e.evaluate(solution))
