import numpy as np
import json
from abc import ABC, abstractmethod
import os
import pyvoronoi
import matplotlib.pyplot as plt

BASE_SEED = 129348


class BaseGenerator(ABC):

    def __init__(self, seed: int, path: str, name: str) -> None:
        """
        seed:
            seed for generators
        path:
            path to directory where graphs should be saved
        name:
            name identifying different graph generators
        """
        self.seed = seed
        self.path = path
        self.name = name
        self.rng = np.random.default_rng(self.seed)

        # global deafult kwargs values(should not be used but is here just in case)
        self.DEF_TIME = 5
        self.DEF_COST = 10
        self.DEF_STATION_COST = 10
        self.DEF_MAX_COST = 30

    @abstractmethod
    def generate(self, size:int | list[int], path: str = None, *args, **kwargs) -> list[str]:
        """ basic generator

        size:
            int - generate one graph with size 'size'
            list - generate graphs with sizes from list 'size'
        path:
            if specified overrides default generator path
        kwargs:
            time: time per kilometer
            cost: cost per kilometer
            station_cost: cost per station
            max_cost: maximum cost

        return:
            list of paths to generated files
        """
        pass

    def _generate_init(self, size:int | list[int], path: str = None, *args, **kwargs) -> None:
        if type(size) is list:
            for ssize in size:
                end_list = []
                end_list.extend(self.generate(ssize, path, **kwargs))
            return end_list

        if path is None:
            path = os.path.join(self.path, f"{self.name}_{size}.json")
        else:
            path += f"_{size}.json"

        self.set_default_kwargs(kwargs.get('time', self.DEF_TIME),
                                kwargs.get('cost', self.DEF_COST),
                                kwargs.get('station_cost', self.DEF_STATION_COST),
                                kwargs.get('max_cost', self.DEF_MAX_COST))
        
        return path

    def generate_batch(self, bid: int | str, size: int | list[int], kwargs_set: list[dict], *args, **kwargs) -> list[str]:
        """ batch generator
        
        bid:
            batch id used in file name to distinguish type of graphs
        size:
            int - generate one graph with size 'size'
            list - generate graphs with sizes from list 'size'
        path:
            if specified overrides default generator path
        kwargs_set: list[dict]
            Every list entry specifies kwargs dict passed to generate.

        return:
            list of paths to generated files
        """
        
        path_sufix = bid + "_" if type(bid) is str else ""
        path_sufix_id = bid if type(bid) is int else 0
        end_files = []

        for kset in kwargs_set:
            cpath = os.path.join(self.path, f"{self.name}_{path_sufix}{path_sufix_id}")
            end_files.extend(self.generate(size, cpath, **kset))
            path_sufix_id += 1

        return end_files
    
    def save_json(self, graph: dict, path: str):
        json_data = {}
        json_data['graph'] = graph
        json_data['metro'] = {'time/km': self.time,
                              'cost/km': self.cost,
                              'cost/station': self.station_cost}
        json_data['max_cost'] = self.max_cost
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(json_data, f)

    def set_default_kwargs(self, time: int, cost: int, station_cost: int, max_cost: int) -> None:
        self.time = time
        self.cost = cost
        self.station_cost = station_cost
        self.max_cost = max_cost


class GridGenerator(BaseGenerator):
    """
    kwargs:
        d: float = 0.5
            randomness parameter of points placement. If 1 a point can be place one grid cell away
            from it's original location. If <= 0 the point will not have any randomness. scales accordingly.
        s: float = 200
            cell size
        c: tuple(int) = (5, 10)
        P: float = 0.9
            dropout rate. specifies a chance that and adjecency will stay in grid.
    """

    def __init__(self, seed: int = BASE_SEED, path: str | None = None) -> None:
        if path is None:
            path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(seed=seed, path=path, name="GridGenerator")

        # basic kwargs default values
        self.DEF_TIME = 5
        self.DEF_COST = 10
        self.DEF_STATION_COST = 10
        self.DEF_MAX_COST = 30

        # custom kwargs default values
        self.DEF_D = 0.5
        self.DEF_S = 200
        self.DEF_C = (5, 10)
        self.DEF_P = 0.9

    @staticmethod
    def _get_neigbors(row, col, width, height) -> list[int]:
            n = []
            
            if row != height-1:
                n.append((row+1)*height + col)
            if row != 0:
                n.append((row-1)*height + col)
            if col != width-1:
                n.append(row*height + col + 1)
            if col != 0:
                n.append(row*height + col - 1)

            return n

    def generate(self, size: int | list[int], path: str = None, *args, **kwargs) -> list[str]:
        """
        size:
            Defines total upper number of created points(the shape is random).
        """

        ret = self._generate_init(size, path, *args, **kwargs)
        if type(ret) == list:
            return ret
        
        self.d = kwargs.get('d', self.DEF_D)
        self.s = kwargs.get('s', self.DEF_S)
        self.c = kwargs.get('c', self.DEF_C)
        self.p = kwargs.get('p', self.DEF_P)

        # deduct the shape
        max_width = int(size)
        width = int(self.rng.integers(1, max_width + 1))
        height = size // width

        # gen graph data
        graph = {}
        graph['nodes'] = height*width
        graph['edges'] = height*(width-1) + (height-1)*width

        for col in range(width):
            for row in range(height):
                idx = row*height + col
                graph[idx] = {}
                angle = np.random.uniform(0, 2 * np.pi)
                radius = float(np.random.uniform(0, self.p*self.s))
                graph[idx]['x'] = col*self.s + radius * float(np.cos(angle))
                graph[idx]['y'] = row*self.s + radius * float(np.sin(angle))
                graph[idx]['adj'] = []
                for neighbor_idx in self._get_neigbors(row, col, width, height):
                    if self.rng.random() < self.p:
                        graph[idx]['adj'].append((neighbor_idx, int(self.rng.integers(self.c[0], self.c[1]))))

        self.save_json(graph, ret)
        return [self.path]


class ConsecutiveGenerator(BaseGenerator):

    def __init__(self, seed: int = BASE_SEED, path: str | None = None) -> None:
        if path is None:
            path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(seed=seed, path=path, name="ConsecutiveGenerator")

        # basic kwargs default values
        self.DEF_TIME = 5
        self.DEF_COST = 10
        self.DEF_STATION_COST = 10
        self.DEF_MAX_COST = 30

    def generate(self, size: int | list[int], path: str = None, *args, **kwargs) -> list[str]:
        """
        size:
            Defines number of consecutive points
        """
        
        ret = self._generate_init(size, path, *args, **kwargs)
        if type(ret) == list:
            return ret
        
        # gen graph data
        graph = {}
        for i in range(size):
            graph['nodes'] = size
            graph['edges'] = size - 1
            graph[i] = {}
            graph[i]['x'] = i
            graph[i]['y'] = i % 2
            graph[i]['adj'] = []
            if i < size - 1:
                graph[i]['adj'].append((i + 1, np.random.randint(5, 10)))
            if i > 0:
                graph[i]['adj'].append((i - 1, np.random.randint(5, 10)))

        self.save_json(graph, ret)
        return [self.path]

# voronoi graphs generator?
# fractals graphs generator?
# https://www.tmwhere.com/city_generation.html
# l systems generators

if __name__ == '__main__':
    cg = GridGenerator(path=os.path.abspath("../benchmark/test"))
    cg.generate_batch("tmp", [16, 23], [{'s': 100}, {'s': 300}])
