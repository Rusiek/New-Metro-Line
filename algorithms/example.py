import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random
import tqdm
import imageio.v2 as imageio
from inspect import signature
from copy import deepcopy


def dist(A, B, G):
    metro_dist_x = G.nodes[A]['x'] - G.nodes[B]['x']
    metro_dist_y = G.nodes[A]['y'] - G.nodes[B]['y']
    return np.sqrt(metro_dist_x ** 2 + metro_dist_y ** 2)


def get_score_cost(G, metro_params, solution, only_cost=False):
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
    
    if only_cost:
        return None, total_cost

    for node_A in G.nodes:
        for _, value in nx.shortest_path_length(G, node_A, weight='weight').items():
            score += value
    
    return score, total_cost


class BaseAlgo:
    def __init__(self, G, metro_params, algo_params, max_cost=float('inf'), vis_path=None, sol_path=None, gif_path=None):
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
        self.actual_scores = None
        self.gif_path = gif_path
        self.metro_params['max_cost'] = max_cost
        self.num_initial_candidates = algo_params['num_initial_candidates']
        self.num_new_candidates = algo_params['num_new_candidates']
        self.randomness_factor = algo_params['randomness_factor']
        self.min_w = algo_params['min_w']
        self.max_w = algo_params['max_w']

    def generate_init_candidates(self) -> list:
        output = []
        while len(output) < self.num_initial_candidates:
            tmp = [i for i in range(self.nodes)]
            random.shuffle(tmp)
            tmp = tmp[:random.randint(2, self.nodes)]
            metro_cost = get_score_cost(self.G, self.metro_params, tmp)[1]
            if metro_cost < self.metro_params['max_cost']:
                output.append(tmp)
        return output

    def generate_new_candidates(self, candidates: list, scores: list) -> list | None:
        ...
 
    def visualize(self, save_plot=False, file_path=None, title=None, **kwargs):
        plt.figure(figsize=(10, 10))
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=self.min_w, vmax=self.max_w)
        
        if title:
            plt.title(title)

        for node in self.G.nodes:
            plt.scatter(self.G.nodes[node]['x'], self.G.nodes[node]['y'], color='black')
        
        for edge in self.G.edges:
            u, v = edge
            plt.plot([self.G.nodes[u]['x'], self.G.nodes[v]['x']], [self.G.nodes[u]['y'], self.G.nodes[v]['y']], color=cmap(norm(self.G.get_edge_data(u, v)['weight'])))
        
        if self.best_solution:
            for i in range(len(self.best_solution) - 1):
                u, v = self.best_solution[i], self.best_solution[i + 1]
                plt.plot([self.G.nodes[u]['x'], self.G.nodes[v]['x']], [self.G.nodes[u]['y'], self.G.nodes[v]['y']], color='green', linewidth=5)

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
                flag = self.generate_init_candidates() if i == 0 else self.generate_new_candidates(self.actual_population, self.actual_scores)
            if not flag:
                break
            self.actual_scores = []
            for candidate in self.actual_population:
                G = self.G.copy()
                for _ in range(len(candidate) - 1):    
                    weight = dist(candidate[0], candidate[1], G) * self.metro_params['time/km']
                    G.add_edge(candidate[0], candidate[1], weight=weight)
                self.actual_scores.append(get_score_cost(G, self.metro_params, candidate)[0])

            best_sol_idx = self.actual_scores.index(min(self.actual_scores))
            best_sol = self.actual_population[best_sol_idx]
            best_score = self.actual_scores[best_sol_idx]
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
        
        self.current_best_score = None
        self.current_best_solution = None
        if visualize:
            self.visualize(save_plot=True, file_path=self.vis_path + 'end',
                            title = f"Iteration: {i}\nBest Score: {self.best_score}\nCurrent Score: {self.current_best_score}")

        if save_best:
            with open(self.sol_path + 'end.sol', 'w') as f:
                f.write(str(self.best_solution))

        if generate_gif:
            self.generate_gif(self.vis_path)

    def generate_gif(self, path):
        import os
        images = sorted(os.listdir(path))
        images_path = deepcopy(images)
        images_gif = []
        for img in images_path:
            images_gif.append(imageio.imread(path + '/' + img))
        imageio.mimsave(self.gif_path, images_gif, fps=1)


class UselessAlgo(BaseAlgo):
    def generate_new_candidates(self, candidates, scores):
        return self.generate_init_candidates()


class BeesAlgo(BaseAlgo):
    def generate_init_candidates(self) -> list:
        output = []
        while len(output) < self.num_initial_candidates:
            u, v = random.sample(range(self.nodes), 2)
            while u == v or (u, v) in output or (v, u) in output:
                u, v = random.sample(range(self.nodes), 2)
            metro_cost = get_score_cost(self.G, self.metro_params, [u, v])[1]
            if metro_cost < self.metro_params['max_cost']:
                output.append([u, v])
        return output


    def generate_new_candidates(self, candidates, scores):
        g_scores = np.array(scores)
        g_scores = 1 / g_scores
        g_scores = np.power(g_scores, self.randomness_factor)
        g_scores = g_scores / g_scores.sum()
        output = []
        it = 0
        while len(output) < self.num_new_candidates and it < 1000:
            tmp = deepcopy(candidates[np.random.choice(range(len(candidates)), p=g_scores)])
            new_vertex = random.randint(0, self.nodes - 1)
            if len(tmp) == self.nodes:
                return output
            while new_vertex in tmp:
                new_vertex = random.randint(0, self.nodes - 1)
            tmp.append(new_vertex)
            metro_cost = get_score_cost(self.G, self.metro_params, tmp, only_cost=True)[1]
            if metro_cost <= self.metro_params['max_cost'] and tmp not in output:
                output.append(tmp)
            it += 1

        return output if len(output) > 0 else None


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

        return G, graph['generator']['min_w'], graph['generator']['max_w'], json_data['metro'], json_data['max_cost']

    G, min_w, max_w, metro_params, max_cost = load_graph('/home/piotr/stdia/BO_SEM6/New-Metro-Line/benchmark/test/ClustersGridGenerator_tmp_1_60.json')
    # metro_params = {'time/km': 0.1, 'cost/km': 10, 'cost/station': 10}
    algo_params = {'num_initial_candidates': 200, 'num_new_candidates': 2000, 'randomness_factor': 1.1, 'min_w': min_w, 'max_w': max_w}
    algo = BeesAlgo(G, metro_params, algo_params, max_cost=max_cost, vis_path='/home/piotr/stdia/BO_SEM6/New-Metro-Line/vis/', sol_path='/home/piotr/stdia/BO_SEM6/New-Metro-Line/sol/', gif_path='solution.gif')
    algo.run(iterations=100, visualize=True, save_best=True, generate_gif=True, verbose=0)
    print(algo.best_solution)
