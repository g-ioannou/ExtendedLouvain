from louvain_modules import visualizer
from louvain_modules.initializer import Initializer
from louvain_modules.visualizer import Visualizer


def setup():
    initializer = Initializer()
    spark = initializer.initialize_spark()
    graph = initializer.initialize_graph()
    k_depth_motif = initializer.k_depth_motif
    visualizer = Visualizer()

    return spark, graph, k_depth_motif, visualizer


def main():
    spark,G, k_depth_motif, visualizer = setup()

    # TODO setup main loop

    G.calculate_cosine_similarities()

    # k_depth_subgraphs = G.find_k_depth_neighbors(k_depth_motif)

    nodes = G.communities()
    
    nodes.show()
    # k_depth_subgraphs.orderBy("e0.cosine").show(k_depth_subgraphs.count())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
