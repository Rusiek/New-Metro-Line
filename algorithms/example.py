import numpy as np
import networkx as nx


def dist(A, B, G):
    metro_dist_x = G.nodes[A]['x'] - G.nodes[B]['x']
    metro_dist_y = G.nodes[A]['y'] - G.nodes[B]['y']
    return np.sqrt(metro_dist_x ** 2 + metro_dist_y ** 2)


def get_score_cost(G, metro_params, solution):
    score = 0
    total_cost = 0
    G = G.copy()

    for i in range(len(solution) - 1):
        distance = dist(solution[i], solution[i + 1], G)
        total_cost += distance * metro_params['cost/km']
        new_weigth = distance * metro_params['time/km']

        G.add_edge(solution[i], solution[i + 1], weight=new_weigth)
        G.add_edge(solution[i + 1], solution[i], weight=new_weigth)
        
        total_cost += len(solution) * metro_params['cost/station']
    
    for node_A in G.nodes:
        for node_B in G.nodes:
            score += nx.shortest_path_length(G, node_A, node_B, weight='weight')
    
    return score, total_cost


class BaseAlgo:
    def __init__(self, G, metro_params, vis_path=None, sol_path=None):
        self.G = G
        self.metro_params = metro_params
        self.nodes = len(G.nodes)
        self.vis_path = vis_path
        self.sol_path = sol_path
        self.actual_best = None
        self.actual_population = None

    def generate_init_candidates(self) -> list: # or list of lists ??
        ...

    def generate_new_candidates(self, candidates: list) -> list: # or list of lists ??
        ...

    def iterate(self, candidates: list) -> list: # or list of lists ??
        ...

    def reduce_curr_population(self, candidates: list, scores: list, ratio: float = 0.1) -> list: # or list of lists ??
        ...

    def visualize(self):
        ...

    def run(self, iterations=100, visualize=False, save_best=False, verbose=0):
        if visualize and self.vis_path is None:
            raise ValueError("Visualization path is not provided")
        if save_best and self.sol_path is None:
            raise ValueError("Solution path is not provided")
        
        for i in range(iterations):
            self.actual_population = \
                self.generate_init_candidates() if i == 0 else self.generate_new_candidates(self.actual_population)
            self.actual_population = self.iterate(self.actual_population)
            scores = [get_score_cost(self.G, metro_params, candidate)[0] for candidate in self.actual_population]
            self.actual_population = self.reduce_curr_population(self.actual_population, scores)
            if visualize:
                self.visualize()

        if save_best:
            best = self.actual_population[0]
            with open(self.sol_path, 'w') as f:
                f.write(str(best))
            self.actual_best = best
        if verbose == 1:
            scores_cost = [get_score_cost(self.G, metro_params, candidate) for candidate in self.actual_population]
            print(scores_cost)


class UselessAlgo(BaseAlgo):
    def generate_init_candidates(self):
        return [[0, 2] for _ in range(10)]

    def generate_new_candidates(self, candidates):
        return candidates

    def iterate(self, candidates):
        return candidates

    def reduce_curr_population(self, candidates, scores, ratio=0.1):
        return candidates

    def visualize(self):
        pass


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
    metro_params = {'time/km': 2, 'cost/km': 10, 'cost/station': 10}
    algo = UselessAlgo(G, metro_params)
    algo.run(iterations=5, verbose=1)
    print(algo.actual_best)
