import networkx as nx
import matplotlib.pyplot as plt
from inspect import signature


class GraphVisualization:

    def __init__(self):
        self.edges = []
        self.weight_edges = []
        self.metro_edges = []
        self.metro_weight_edges = []

    def add_weight_edge(self, u, v, w):
        self.weight_edges.append([u, v, w])

    def add_edge(self, u, v):
        self.edges.append([u, v])

    def add_metro_edge(self, u, v):
        self.metro_edges.append([u, v])

    def add_weight_metro_edge(self, u, v, w):
        self.metro_weight_edges.append([u, v, w])

    def prepare_normal_data(self,
                            G,
                            edge_color="black",
                            edge_style="solid"
                            ):
        G.add_edges_from(self.edges, edge_color=edge_color, edge_style=edge_style)
        G.add_weighted_edges_from(self.weight_edges, edge_color=edge_color, edge_style=edge_style)

    def prepare_metro_data(self,
                           G,
                           metro_edge_color="black",
                           metro_edge_style="dashed",
                           ):

        G.add_edges_from(self.metro_edges, edge_color=metro_edge_color, edge_style=metro_edge_style)
        G.add_weighted_edges_from(self.metro_weight_edges, edge_color=metro_edge_color, edge_style=metro_edge_style)

    def visualize(self, save_plot=False, file_path=None, **kwargs):
        G = nx.DiGraph()

        valid_normal_kwargs = signature(GraphVisualization.prepare_normal_data).parameters.keys()
        valid_metro_kwargs = signature(GraphVisualization.prepare_metro_data).parameters.keys()
        valid_kwargs = (valid_normal_kwargs | valid_metro_kwargs)
        if any(k not in valid_kwargs for k in kwargs):
            invalid_args = ", ".join([k for k in kwargs if k not in valid_kwargs])
            raise ValueError(f"Received invalid argument(s): {invalid_args}")

        normal_kwargs = {k: v for k, v in kwargs.items() if k in valid_normal_kwargs}
        metro_kwargs = {k: v for k, v in kwargs.items() if k in valid_metro_kwargs}
        self.prepare_normal_data(G, **normal_kwargs)
        self.prepare_metro_data(G, **metro_kwargs)
        all_edge_colors = [G[u][v]["edge_color"] for u, v in G.edges]
        all_edge_styles = [G[u][v]["edge_style"] for u, v in G.edges]

        pos = nx.drawing.spiral_layout(G)
        nx.draw_networkx(G, pos, edge_color=all_edge_colors, style=all_edge_styles)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
        if save_plot:
            if not file_path:
                raise Exception("File path not specified")
            plt.savefig(file_path)
        plt.show()


if __name__ == '__main__':

    G = GraphVisualization()
    G.add_edge(0, 2)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(5, 3)
    G.add_edge(3, 4)
    G.add_edge(1, 0)
    G.add_weight_edge(4, 0, 20)
    G.add_metro_edge(0, 3)
    G.add_weight_metro_edge(5, 0, 2)
    G.visualize(
        edge_color="green",
        metro_edge_color="blue",
        edge_style="solid",
        metro_edge_style="dashed",
        # save_plot=True,
        # file_path="g1.png"
    )
