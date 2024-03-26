import sys
from algorithms.example import Example
from benchmark.evaluate import Evaluate
from loader import load_graph, load_constraints, load_metro_params

if __name__ == '__main__':
    dataset = sys.argv[1]
    G = load_graph(dataset)
    metro = load_metro_params(dataset)
    max_cost = load_constraints(dataset)
    worker = Evaluate(G, metro, max_cost)
    solution = Example(G).find_solution()
    output = worker.evaluate(solution)
    print(output)
