class Example:
    def __init__(self, G):
        self.G = G
        self.nodes = len(G.nodes)
    
    def find_solution(self):
        return [0, self.nodes - 1]
