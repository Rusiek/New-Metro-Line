import sys
import os
from algorithms.example import UselessAlgo, BeesAlgo
from benchmark.evaluate import Evaluate
from loader import load_graph, load_constraints, load_metro_params
from generator.generator import GridGenerator
import tqdm

if __name__ == '__main__':
    dataset = sys.argv[1]
    G = load_graph(dataset)
    metro = load_metro_params(dataset)
    max_cost = load_constraints(dataset)
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

    algo = BeesAlgo(G, metro, max_cost=max_cost, vis_path=vis_path, sol_path=sol_path, gif_path=gif_path)
    algo.run(visualize=True, save_best=True, generate_gif=True, verbose=0)
    solution = algo.best_solution
    output = worker.evaluate(solution)
    print(output)

# -------------------------- GENERATE THAN CALCULATE ------------------------- #

def calc(dset):
    print(f"Handling dataset: {dset}")
    dataset = dset
    G = load_graph(dataset)
    metro = load_metro_params(dataset)
    max_cost = load_constraints(dataset)
    worker = Evaluate(G, metro, max_cost)

    vis_path = dset.split('.')[0]
    try:
        os.mkdir(vis_path)
    except FileExistsError:
        pass

    vis_path += '/vis/'
    try:
        os.mkdir(vis_path)
    except FileExistsError:
        pass

    sol_path = dset.split('.')[0]
    try:
        os.mkdir(sol_path)
    except FileExistsError:
        pass

    sol_path += '/sol/'
    try:
        os.mkdir(sol_path)
    except FileExistsError:
        pass

    gif_path = dset.split('.')[0] + '/solution.gif'

    algo = BeesAlgo(G, metro, max_cost=max_cost, vis_path=vis_path, sol_path=sol_path, gif_path=gif_path)
    algo.run(iterations=40, visualize=True, save_best=True, generate_gif=True, verbose=0)
    solution = algo.best_solution
    output = worker.evaluate(solution)
    print(output)

# if __name__ == '__main__':

#     cg = GridGenerator(path=os.path.abspath("benchmark/test"))
#     flist = cg.generate_batch("BigTest", 60)

#     for f in tqdm.tqdm(flist):
#         calc(f)
