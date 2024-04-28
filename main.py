import sys
from algorithms.example import UselessAlgo
from benchmark.evaluate import Evaluate
from loader import load_graph, load_constraints, load_metro_params

if __name__ == '__main__':
    dataset = sys.argv[1]
    G = load_graph(dataset)
    metro = load_metro_params(dataset)
    max_cost = load_constraints(dataset)
    worker = Evaluate(G, metro, max_cost)
    algo = UselessAlgo(G, metro, vis_path='algorithms/visualizations/vis', sol_path='algorithms/solutions/sol')
    algo.run(visualize=True, save_best=True, generate_gif=True, verbose=0)
    solution = algo.best_solution
    output = worker.evaluate(solution)
    print(output)
