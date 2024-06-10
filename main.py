import sys
import os
from algorithms.example import BeesAlgo
from benchmark.evaluate import Evaluate
from loader import load_graph, load_constraints, load_metro_params, load_generator_data

if __name__ == '__main__':
    dataset = sys.argv[1]
    G = load_graph(dataset)
    metro = load_metro_params(dataset)
    max_cost = load_constraints(dataset)
    min_w, max_w = load_generator_data(dataset)
    worker = Evaluate(G, metro, max_cost)

    vis_path = sys.argv[1].split('.')[0]
    try:
        os.mkdir(vis_path)
    except FileExistsError:
        pass

    vis_path += '/vis/'
    try:
        os.mkdir(vis_path)
    except FileExistsError:
        pass

    sol_path = sys.argv[1].split('.')[0]
    try:
        os.mkdir(sol_path)
    except FileExistsError:
        pass

    sol_path += '/sol/'
    try:
        os.mkdir(sol_path)
    except FileExistsError:
        pass

    gif_path = sys.argv[1].split('.')[0] + '/solution.gif'

    algo_params = {'num_initial_candidates': 100, 'num_new_candidates': 1000, 'randomness_factor': 1, 'min_w': min_w, 'max_w': max_w}
    algo = BeesAlgo(G, metro, algo_params, max_cost=max_cost, vis_path=vis_path, sol_path=sol_path, gif_path=gif_path)
    algo.run(visualize=True, save_best=True, generate_gif=True, verbose=0)
    solution = algo.best_solution
    output = worker.evaluate(solution)
    print(output)
