from pyvis.network import Network
from louvain_modules.obj import Graph


class Visualizer:
    @staticmethod
    def visualize(G:Graph):
        nt = Network("100%","100%",heading="Graph")
        nt.from_nx(G.to_nx())
        nt.toggle_physics(False)
        nt.save_graph("test.html")