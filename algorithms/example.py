import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random
import tqdm
import imageio.v2 as imageio
from inspect import signature
from copy import deepcopy
import random


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
        self.randomness_factor = algo_params.get('randomness_factor', 1)
        self.min_w = algo_params['min_w']
        self.max_w = algo_params['max_w']
        self.stagnated_generations = 0
        self.stagnation_limit = algo_params.get('stagnation_limit', 3)
        self.elite_fraction = algo_params.get('elite_fraction', 0.1)
        self.mutation_rate = algo_params.get('mutation_rate', 0.5)

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

        progress_bar = tqdm.tqdm(range(iterations))

        for i in progress_bar:
            self.actual_population = \
                flag = self.generate_init_candidates() if i == 0 else self.generate_new_candidates(self.actual_population, self.actual_scores)
            if not flag:
                tqdm.tqdm.close(progress_bar)
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
                self.stagnated_generations = 0
            else:
                self.stagnated_generations += 1

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


class CSOAlgo(BaseAlgo):
    def __init__(self, G, metro_params, algo_params,
                 visual_range,
                 inertia_coefficient,
                 step,
                 max_cost,
                 vis_path,
                 sol_path,
                 gif_path
                 ):
        super().__init__(G, metro_params, algo_params, max_cost=max_cost,
                 vis_path=vis_path,
                 sol_path=sol_path,
                 gif_path=gif_path)
        self.visual_range = visual_range
        self.inertia_coefficient = inertia_coefficient
        self.step = step

    def generate_init_candidates(self) -> list:
        output = []
        while len(output) < self.num_initial_candidates:
            tmp = [i for i in range(self.nodes)]
            random.shuffle(tmp)
            tmp = tmp[:random.randint(2, 3)]
            metro_cost = get_score_cost(self.G, self.metro_params, tmp)[1]
            if metro_cost < self.metro_params['max_cost']:
                output.append(tmp)
        return output

    def generate_new_candidates(self, candidates: list, scores: list) -> list | None:
        Pg = None
        Pg_score = None
        Pi_list = []
        def recalculate_best_score(candidate):
            nonlocal Pg, Pg_score
            can_score = get_score_cost(self.G, self.metro_params, candidate)[0]
            if not Pg or Pg_score > can_score:
                Pg = candidate
                Pg_score = can_score

        for candidate in candidates:
            Pi = candidate
            recalculate_best_score(candidate)

            for second_candidate in candidates:
                first_can_set = set(candidate)
                second_can_set = set(second_candidate)
                if len(first_can_set - second_can_set) < self.visual_range:
                    recalculate_best_score(second_candidate)

            Pi_list.append(Pi)

        for candidate, Pi in zip(candidates, Pi_list):

            if not random.randint(0, 1):
                continue

            currBest = Pi
            if candidate == Pi:
                currBest = Pg
            stops = list(set(currBest) - set(candidate))
            for curr_step in range(min(self.step, len(stops))):
                cost_if_added = get_score_cost(self.G, self.metro_params, candidate + [stops[curr_step]], True)[1]
                if cost_if_added < self.metro_params['max_cost']:
                    candidate.append(stops[curr_step])

            recalculate_best_score(candidate)

        candidate = random.choice(candidates)
        candidate[random.randint(0, len(candidate) - 1)] = random.randint(1, self.nodes - 1)

        recalculate_best_score(candidate)

        candidate_id = random.choice([i for i in range(0, len(candidates))])

        if random.randint(0, 1):
            candidates[candidate_id] = Pg.copy()

        else:
            candidates[candidate_id] = [candidates[candidate_id][-1]]
        recalculate_best_score(candidate)

        return candidates


class GeneticAlgo(BaseAlgo):

    def generate_init_candidates(self) -> list:
        output = []
        while len(output) < self.num_initial_candidates:
            nodes = list(G.nodes)
            random.shuffle(nodes)

            random_path = []
            metro_cost = 0
            while nodes and metro_cost < self.metro_params['max_cost']:
                element = nodes.pop(0)
                random_path.append(element)
                metro_cost = get_score_cost(self.G, self.metro_params, random_path, only_cost=True)[1]

            random_path.pop()
            output.append(random_path)

        return output

    def generate_new_candidates(self, candidates, scores):
        new_candidates = []

        def crossover(parent1, parent2):
            crossover_point = random.randint(0, min(len(parent1), len(parent2)))
            child1 = parent1[:crossover_point] + [x for x in parent2 if x not in parent1[:crossover_point]]
            child2 = parent2[:crossover_point] + [x for x in parent1 if x not in parent2[:crossover_point]]
            return child1, child2

        def mutate(candidate):
            while random.random() < self.mutation_rate:
                available_nodes = list(set(G.nodes) - set(candidate))
                candidate_index = random.sample(range(len(candidate)), 1)[0]
                new_node = random.sample(available_nodes, 1)[0]

                candidate[candidate_index] = new_node

            return candidate

        sorted_candidates = [candidate for _, candidate in sorted(zip(scores, candidates))]
        elite_count = int(self.elite_fraction * len(sorted_candidates))
        elite_candidates = sorted_candidates[:elite_count]

        it = 0
        while len(new_candidates) < self.num_new_candidates:
            parent1, parent2 = random.sample(elite_candidates, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)

            for child in [child1, child2]:
                metro_cost = get_score_cost(self.G, self.metro_params, child, only_cost=True)[1]
                if it > self.num_new_candidates * 10:
                    while metro_cost > self.metro_params['max_cost']:
                        child.pop()
                        metro_cost = get_score_cost(self.G, self.metro_params, child, only_cost=True)[1]

                if metro_cost <= self.metro_params['max_cost'] and child not in new_candidates:
                    new_candidates.append(child)
            it += 1

        if self.stagnated_generations == self.stagnation_limit:
            return None

        return new_candidates if new_candidates else None


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

    G, min_w, max_w, metro_params, max_cost = load_graph('C:\\Users\\Wojtek\\PycharmProjects\\BO\\New-Metro-Line\\generator\\benchmark\\test\\ClustersGridGenerator_tmp_1_60.json')
    # metro_params = {'time/km': 0.1, 'cost/km': 10, 'cost/station': 10}
    algo_params = {'num_initial_candidates': 200, 'num_new_candidates': 2000, 'randomness_factor': 1.1, 'min_w': min_w, 'max_w': max_w}
    algo = BeesAlgo(G, metro_params, algo_params, max_cost=max_cost, vis_path='C:\\Users\\Wojtek\\PycharmProjects\\BO\\New-Metro-Line\\vis\\', sol_path='C:\\Users\\Wojtek\\PycharmProjects\\BO\\New-Metro-Line\\sol\\', gif_path='solution.gif')
    # G, min_w, max_w, metro_params, max_cost = load_graph('ClustersGridGenerator_tmp_0_60.json')
    # algo_params = {'num_initial_candidates': 200, 'num_new_candidates': 200, 'min_w': min_w, 'max_w': max_w,
    #                'stagnation_limit': 4, 'elite_fraction': 0.2, 'mutation_rate': 0.25}
    # algo = GeneticAlgo(G, metro_params, algo_params, max_cost=max_cost, vis_path='vis/', sol_path='sol/',
    #                    gif_path='solution.gif')
    algo.run(iterations=100, visualize=True, save_best=True, generate_gif=True, verbose=0)
    print(algo.best_solution)

    # algo = CSOAlgo(G, metro_params, algo_params, 2, 2, 1,
    #                max_cost=max_cost,
    #                vis_path='C:\\Users\\Wojtek\\PycharmProjects\\BO\\New-Metro-Line\\vis\\',
    #                sol_path='C:\\Users\\Wojtek\\PycharmProjects\\BO\\New-Metro-Line\\sol\\',
    #                gif_path='solution.gif')
    # algo.run(iterations=15, visualize=True, save_best=True, generate_gif=True, verbose=0)
