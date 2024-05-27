import numpy as np
import json
from abc import ABC, abstractmethod
import os
from collections import defaultdict

BASE_SEED = 129348


def in_adj(idx: int, adj: list) -> bool:
    for sadj in adj:
        if idx == sadj[0]:
            return True
    return False

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
            end_list = []
            for ssize in size:
                end_list.extend(self.generate(ssize, path, **kwargs))
            return end_list

        if path is None:
            path = os.path.join(self.path, f"{self.name}_{size}.json")
        else:
            path += f"_{size}.json"
        
        return path

    def generate_batch(self, bid: int | str, size: int | list[int], kwargs_set: list[dict] | None = None, *args, **kwargs) -> list[str]:
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

        if kwargs_set is None:
            if isinstance(size, int):
                kwargs_set = [{}]
            else:
                kwargs_set = [{} for _ in range(len(size))]
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


class GridGenerator(BaseGenerator):
    """
    kwargs:
        d: float = 0.2
            randomness parameter of points placement. If 1 a point can be place one grid cell away
            from it's original location. If <= 0 the point will not have any randomness. scales accordingly.
        s: float = 10
            cell size
        c: float = 30
            The higher the value the slower the streat traffic flows. The real flow value is randomly choosen
            from range (c, inf) with poisson distribution. The weight is calculated with equation w=dist*flow
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
        self.DEF_STATION_COST = 50
        self.DEF_MAX_COST = 2000

        # custom kwargs default values
        self.cosparam = 1
        self.DEF_D = 0.4
        self.DEF_S = 10
        self.DEF_C = 27
        self.DEF_P = 0.9

    @staticmethod
    def _get_neigbors(row, col, width, height) -> list[int]:
            n = []
            
            if row != height-1:
                n.append((row+1)*width + col)
            if row != 0:
                n.append((row-1)*width + col)
            if col != width-1:
                n.append(row*width + col + 1)
            if col != 0:
                n.append(row*width + col - 1)

            return n
    
    @staticmethod
    def _get_diagonal(maxx, maxy, minx, miny):
        return np.sqrt((maxx-minx)**2 + (maxy-miny)**2)

    def generate(self, size: int | list[int], path: str = None, *args, **kwargs) -> list[str]:
        """
        size:
            Defines total upper number of created points(the shape is random).
        """

        ret = self._generate_init(size, path, *args, **kwargs)
        if type(ret) == list:
            return ret
        
        _nid_map = {}
        _nfid = 0
        def get_id(id):
            nonlocal _nfid
            try:
                return _nid_map[id]
            except KeyError:
                nid = _nfid
                _nfid += 1
                _nid_map[id] = nid
                return nid
            
        maxx, maxy, minx, miny = None, None, None, None
        def update_lims(x, y):
            nonlocal maxx, maxy, minx, miny
            if maxx is None or x > maxx:
                maxx = x
            if maxy is None or y > maxy:
                maxy = y
            if minx is None or x < minx:
                minx = x
            if miny is None or y < miny:
                miny = y
        
        self.d = kwargs.get('d', self.DEF_D)
        self.s = kwargs.get('s', self.DEF_S)
        self.c = kwargs.get('c', self.DEF_C)
        self.p = kwargs.get('p', self.DEF_P)

        # deduct the shape
        shapes = set()
        for w in range(max(int(np.sqrt(size)-3), 1), size//2+1):
            h = size//w
            shapes.add((w, h))
        width, height = list(shapes)[self.rng.integers(0, len(shapes))]

        # gen graph data
        graph = defaultdict(lambda: defaultdict(list))
        graph['nodes'] = height*width
        edge_count = 0
        min_w, max_w = None, None
        def update_minmax(nv):
            nonlocal min_w, max_w
            if min_w is None:
                min_w = nv
            if max_w is None:
                max_w = nv
            if nv < min_w:
                min_w = nv
            if nv > max_w:
                max_w = nv

        for row in range(height):
            for col in range(width):
                idx = row*width + col
                angle = np.random.uniform(0, 2 * np.pi)
                radius = float(np.random.uniform(0, self.d*self.s))
                x = col*self.s + radius * float(np.cos(angle))
                y = row*self.s + radius * float(np.sin(angle))
                graph[get_id(idx)]['x'] = x
                graph[get_id(idx)]['y'] = y
                update_lims(x, y)

        for row in range(height):
            for col in range(width):
                idx = row*width + col
                for neighbor_idx in self._get_neigbors(row, col, width, height):
                    if self.rng.random() < self.p:
                        
                        dx = graph[get_id(idx)]['x'] - graph[get_id(neighbor_idx)]['x']
                        dy = graph[get_id(idx)]['y'] - graph[get_id(neighbor_idx)]['y']
                        dist = np.sqrt(dx**2 + dy**2)

                        if not in_adj(get_id(neighbor_idx), graph[get_id(idx)]['adj']):
                            rdm_dist = dist*float(self.rng.poisson(3))
                            update_minmax(rdm_dist)
                            graph[get_id(idx)]['adj'].append((get_id(neighbor_idx), rdm_dist))
                            edge_count += 1
                        if not in_adj(get_id(idx), graph[get_id(neighbor_idx)]['adj']):
                            rdm_dist = dist*float(self.rng.poisson(3))
                            update_minmax(rdm_dist)
                            graph[get_id(neighbor_idx)]['adj'].append((get_id(idx), rdm_dist))
                            edge_count += 1

        graph['edges'] = edge_count

        diag = float(self._get_diagonal(maxx, maxy, minx, miny))
        self.time = kwargs.get('time', self.DEF_TIME)
        self.cost = kwargs.get('cost', self.DEF_COST)
        self.station_cost = kwargs.get('station_cost', self.DEF_STATION_COST)
        self.max_cost = kwargs.get('max_cost', diag * self.cost * (1 + self.cosparam) \
            + ((diag)/(self.s*(1+self.cosparam)))*self.station_cost
        )

        graph['generator'] = {'min_w': min_w, 'max_w': max_w}

        self.save_json(graph, ret)
        return [ret]


class ConsecutiveGenerator(BaseGenerator):

    def __init__(self, seed: int = BASE_SEED, path: str | None = None, size: int = 10) -> None:
        if path is None:
            path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(seed=seed, path=path, name="ConsecutiveGenerator")
        self.size = size
        # basic kwargs default values
        self.DEF_TIME = 0.1
        self.DEF_COST = 10
        self.DEF_STATION_COST = 10
        self.DEF_MAX_COST = 30 * self.size

    def generate(self, path: str = None, *args, **kwargs) -> list[str]:
        """
        size:
            Defines number of consecutive points
        """
        
        ret = self._generate_init(self.size, path, *args, **kwargs)
        if type(ret) == list:
            return ret
        
        # gen graph data
        graph = {}
        for i in range(self.size):
            graph['nodes'] = self.size
            graph['edges'] = 2 * self.size
            graph[i] = {}
            graph[i]['x'] = i
            graph[i]['y'] = i % 2
            graph[i]['adj'] = []
            if i < self.size - 1:
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
    cg = GridGenerator(path=os.path.abspath("benchmark/test"))
    cg.generate_batch("tmp", [16, 23], [{'s': 100}, {'s': 300}])

    # cg = ConsecutiveGenerator(path=os.path.abspath("../benchmark/small"), size = 60)
    # cg.generate(path=os.path.abspath("../benchmark/small/consecutive_30.json"))
