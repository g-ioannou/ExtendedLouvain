from pprint import pp, pprint
from louvain_modules.initializer import Initializer
from louvain_modules.visualizer import Visualizer

import logging

def setup():
    initializer = Initializer()
    spark = initializer.initialize_spark()

    k_depth = initializer.k_depth
    graph = initializer.initialize_graph(k_depth=k_depth)

    visualizer = Visualizer()

    COSINE_THRESH = float(initializer.parameters["cosine_threshold"])
    WEIGHT_THRESH = float(initializer.parameters["weight_threshold"])

    return spark, graph, k_depth, COSINE_THRESH, WEIGHT_THRESH, visualizer


def main():
    spark, G, k_depth, COSINE_THRESH, WEIGHT_THRESH, visualizer = setup()

    # TODO setup main loop

    G.initialize_communities()  # todo move this to initializer

    G.find_k_depth_neighbors()
    G.nodes.orderBy("id").show(truncate=False)

    G.calculate_cosine_similarities()
    

    partition = G.get_partition_using_metrics_old(
        weight_threshold=WEIGHT_THRESH, cosine_threshold=COSINE_THRESH, iteration=1
    )
    G.update_with_partition(partition=partition)
    G.calculate_cosine_similarities()
    

    partition = G.get_partition_using_metrics(weight_threshold=WEIGHT_THRESH, cosine_threshold=COSINE_THRESH,iteration=2)
    
    print(partition)

    exit()

    G.update_with_partition(partition=partition)

    # next iteration
    G.calculate_cosine_similarities()
    visualizer.visualize(G, 1)
    exit()

    print("-----------------------------")
    partition = G.get_partition_using_metrics(
        weight_threshold=WEIGHT_THRESH, cosine_threshold=COSINE_THRESH, iteration=2
    )
    pprint(partition)

    G.update_with_partition(partition=partition)
    G.nodes.orderBy("id").show(G.nodes.count())
    G.calculate_cosine_similarities()
    visualizer.visualize(G, 2)
    exit()

    exit()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        logging.exception("An error occured")
        print()
        