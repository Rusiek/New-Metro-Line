import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import tqdm
import imageio.v2 as imageio
from inspect import signature
from copy import deepcopy


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
        self.current_best_solution = None
        self.current_best_score = None
        self.best_solution = None
        self.best_score = None
        self.actual_population = None

    def generate_init_candidates(self, n = 100) -> list: # or list of lists ??
        ...

    def generate_new_candidates(self, candidates: list) -> list: # or list of lists ??
        ...

    def iterate(self, candidates: list) -> list: # or list of lists ??
        ...

    def reduce_curr_population(self, candidates: list, scores: list, ratio: float = 0.1) -> list: # or list of lists ??
        ...
 
    def visualize(self, save_plot=False, file_path=None, title=None, **kwargs):
        plt.figure(figsize=(10, 10))
        
        if title:
            plt.title(title)

        for node in self.G.nodes:
            plt.scatter(self.G.nodes[node]['x'], self.G.nodes[node]['y'], color='black')
        
        for edge in self.G.edges:
            u, v = edge
            plt.plot([self.G.nodes[u]['x'], self.G.nodes[v]['x']], [self.G.nodes[u]['y'], self.G.nodes[v]['y']], color='black')
        
        if self.best_solution:
            for i in range(len(self.best_solution) - 1):
                u, v = self.best_solution[i], self.best_solution[i + 1]
                plt.plot([self.G.nodes[u]['x'], self.G.nodes[v]['x']], [self.G.nodes[u]['y'], self.G.nodes[v]['y']], color='green')

        if self.current_best_solution:
            for i in range(len(self.current_best_solution) - 1):
                u, v = self.current_best_solution[i], self.current_best_solution[i + 1]
                plt.plot([self.G.nodes[u]['x'], self.G.nodes[v]['x']], [self.G.nodes[u]['y'], self.G.nodes[v]['y']], color='blue', linestyle='dashed')

        if save_plot:
            if not file_path:
                raise Exception("File path not specified")
            plt.savefig(file_path)
        plt.close()

    def run(self, iterations=100, visualize=False, save_best=False, generate_gif=False, verbose=0):
        if visualize and self.vis_path is None:
            raise ValueError("Visualization path is not provided")
        if save_best and self.sol_path is None:
            raise ValueError("Solution path is not provided")
        
        for i in tqdm.tqdm(range(iterations)):
            self.actual_population = \
                self.generate_init_candidates() if i == 0 else self.generate_new_candidates(self.actual_population)
            self.actual_population = self.iterate(self.actual_population)
            scores = []
            for candidate in self.actual_population:
                G = self.G.copy()
                for _ in range(len(candidate) - 1):    
                    weight = dist(candidate[0], candidate[1], G) * self.metro_params['time/km']
                    G.add_edge(candidate[0], candidate[1], weight=weight)
                scores.append(get_score_cost(G, self.metro_params, candidate)[0])
            self.actual_population = self.reduce_curr_population(self.actual_population, scores)

            best_sol_idx = scores.index(min(scores))
            best_sol = self.actual_population[best_sol_idx]
            best_score = scores[best_sol_idx]
            self.current_best_solution = best_sol
            self.current_best_score = best_score
            if visualize:
                self.visualize(save_plot=True, file_path=self.vis_path + str(i).zfill(4),
                               title = f"Iteration: {i}\nBest Score: {self.best_score}\nCurrent Score: {self.current_best_score}")

            if self.best_solution is None or best_score < self.best_score:
                self.best_solution = best_sol
                self.best_score = best_score
            
            if save_best:
                with open(self.sol_path + str(i).zfill(4) + '.sol', 'w') as f:
                    f.write(str(best_sol))
        
            if verbose == 1:
                scores_cost = [get_score_cost(self.G, metro_params, candidate) for candidate in self.actual_population]
                print(scores_cost)
            
        if generate_gif:
            self.generate_gif('vis.gif', iterations=iterations)

    def generate_gif(self, path, iterations=100):
        images = []
        for i in range(iterations):
            images.append(imageio.imread(self.vis_path + str(i).zfill(4) + '.png'))
        imageio.mimsave(path, images)


class UselessAlgo(BaseAlgo):
    def generate_init_candidates(self):
        output = []
        for _ in range(10):
            u, v, x = random.sample(range(self.nodes), 3)
            while u == v or v == x or u == x:
                u, v, x = random.sample(range(self.nodes), 3)
            output.append([u, v, x])
        return output

    def generate_new_candidates(self, candidates):
        return self.generate_init_candidates()

    def iterate(self, candidates):
        return candidates

    def reduce_curr_population(self, candidates, scores, ratio=0.1):
        return candidates


if __name__ == '__main__':
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

    G = load_graph('/home/rusiek/Studia/vi_sem/New-Metro-Line/benchmark/test/GridGenerator_tmp_0_16.json')
    metro_params = {'time/km': 0, 'cost/km': 10, 'cost/station': 10}
    algo = UselessAlgo(G, metro_params, vis_path='visualizations/vis', sol_path='solutions/sol')
    algo.run(iterations=100, visualize=True, save_best=True, generate_gif=True, verbose=0)
    # print(algo.best_solution)
