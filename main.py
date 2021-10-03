from louvain_modules import visualizer
from louvain_modules.initializer import Initializer
from louvain_modules.visualizer import Visualizer

from pyspark.sql import functions as F
from pyspark.sql import Window


def setup():
    initializer = Initializer()
    spark = initializer.initialize_spark()
    graph = initializer.initialize_graph()
    k_depth = initializer.k_depth
    visualizer = Visualizer()

    COSINE_THRESH = float(initializer.parameters["cosine_threshold"])
    WEIGHT_THRESH = float(initializer.parameters["weight_threshold"])

    return spark, graph, k_depth, COSINE_THRESH, WEIGHT_THRESH, visualizer


def main():
    spark, G, k_depth, COSINE_THRESH, WEIGHT_THRESH, visualizer = setup()

    # TODO setup main loop

    G.calculate_cosine_similarities()

    G.initialize_communities()  # todo move this to initializer

    G.find_k_depth_neighbors(k_depth)

    G.get_partition_using_metrics(weight_threshold=WEIGHT_THRESH, cosine_threshold=COSINE_THRESH)

    exit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
