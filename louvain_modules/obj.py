from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import subgraph
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from louvain_modules.reader import Reader


from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame
from graphframes import GraphFrame
from pyspark.sql import functions as F
import numpy

import networkx
import pprint


class Graph:
    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame, spark: SparkSession) -> None:

        self.nodes = nodes_df
        self.edges = edges_df

    def modularity(self):

        sum_of_weights = self.edges.groupBy().agg(F.sum(F.col("weight")).alias("sum_of_graph_links"))

        sum_of_weights.show()

    def initialize_communities(self) -> DataFrame:

        self.nodes = (
            self.nodes.select("id", "vector")
            .withColumn("community_nodes", F.array("id"))
            .withColumn("community_edges", F.array())
            .withColumn("community_vectors", F.array("vector"))
            .withColumn("center", F.col("id"))
        )

    def to_nx(self) -> networkx.Graph:

        self._node_attr = (
            self.nodes.drop("community_vectors", "community_edges").toPandas().set_index("id").to_dict("index")
        )

        G_nx = networkx.from_pandas_edgelist(
            self.edges.toPandas(), source="src", target="dst", edge_attr=["weight", "cosine"]
        )

        networkx.set_node_attributes(G_nx, self._node_attr)

        return G_nx

    def filter_out_small_communities(self, min_node_count: int = 10):
        g = GraphFrame(self.nodes, self.edges).dropIsolatedVertices()

        components = g.connectedComponents()

        grouped_components = (
            components.groupBy("component")
            .agg({"component": "count"})
            .filter("count(component)>=" + str(min_node_count))
        )

        self.nodes = (
            components.join(grouped_components, components.component == grouped_components.component, "leftsemi")
            .drop("component")
            .checkpoint()
        )

        self.edges = (
            g.edges.join(self.nodes, g.edges.src == self.nodes.id, "leftsemi")
            .join(self.nodes, g.edges.dst == self.nodes.id, "leftsemi")
            .checkpoint()
        )

    def calculate_cosine_similarities(self):
        @F.udf(FloatType())
        def udf_numpy_cosine(vec1: list, vec2: list):
            inner_product = numpy.inner(vec1, vec2)
            norms_product = numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2)

            if norms_product != 0:
                return float(inner_product / norms_product)
            else:
                return float(0)

        self.edges = (
            self.edges.join(self.nodes, on=self.edges.src == self.nodes.id, how="inner")
            .drop("id")
            .withColumnRenamed("vector", "vector_src")
            .join(self.nodes, on=self.nodes.id == self.edges.dst, how="inner")
            .drop("id")
            .withColumnRenamed("vector", "vector_dst")
            .withColumn("cosine", F.round(udf_numpy_cosine(F.col("vector_src"), F.col("vector_dst")), 3))
            .drop(*["vector_src", "vector_dst"])
        )

    def find_k_depth_neighbors(self, depth: int = 1) -> DataFrame:
        k_depth_neighbors = self.nodes.select("id").withColumn(f"depth_0_neighbor", F.col("id"))

        for i in range(1, depth + 1):
            k_depth_neighbors = k_depth_neighbors.withColumn(f"depth_{i}_neighbor", F.lit(None).cast(IntegerType()))

        for i in range(0, depth):
            k_depth_neighbors = (
                k_depth_neighbors.join(
                    self.edges.select("src", "dst"),
                    on=(
                        (self.edges.src == k_depth_neighbors[f"depth_{i}_neighbor"])
                        | (self.edges.dst == k_depth_neighbors[f"depth_{i}_neighbor"])
                    ),
                    how="left",
                )
                .withColumn(
                    f"depth_{i+1}_neighbor",
                    F.when(F.col(f"depth_{i}_neighbor") == F.col("dst"), F.col("src")).otherwise(F.col("dst")),
                )
                .drop("src", "dst")
            )

        k_depth_neighbors = (
            k_depth_neighbors.withColumn(
                "k_depth_neighbors", F.array([F.col(f"depth_{i}_neighbor") for i in range(0, depth + 1)])
            )
            .groupBy("id")
            .agg(F.array_distinct(F.flatten(F.collect_set("k_depth_neighbors"))).alias("k_depth_neighbors"))
            .withColumn("k_depth_neighbors", F.array_except(F.col("k_depth_neighbors"), F.array(F.col("id"))))
            .withColumnRenamed("id", "id_temp")
        )

        self.nodes = (
            self.nodes.join(
                k_depth_neighbors,
                on=self.nodes.id == k_depth_neighbors.id_temp,
                how="inner",
            )
            .drop("id_temp")
            .select(  # ? THIS IS UNEEDED
                "id",
                "center",
                "community_nodes",
                "community_edges",
                "k_depth_neighbors",
                "community_vectors",
            )
        )

    def get_partition_using_metrics(self, weight_threshold: float, cosine_threshold: float):
        G = self.to_nx()

        communities = self._node_attr
        pprint.pprint(communities)
        from timeit import default_timer as timer

        start = timer()
        for node in communities:
            print("checking node: ", node)
            if len(communities[node]["community_nodes"]) > 1:
                print(node, " is already a community")
                print("----------------")
                continue

            # find all neighbors of node i
            node_neighbors = {
                neighbor: {
                    "center": communities[neighbor]["center"],
                    "community_nodes": communities[neighbor]["community_nodes"],
                }
                for neighbor in G.neighbors(node)
            }
            print("node_neighbors: ", node_neighbors)

            # initialize best community to None
            best_neighboring_community = None
            max_cosine_sim = -1
            max_weight = -1
            max_delta_Q = -1

            merged_by_metrics = False
            for neighbor in node_neighbors:
                edge_weight = G[node][neighbor]["weight"]
                cosine_sim = G[node][neighbor]["cosine"]

                # first check if cosine similarity or edge weight for each neighbor are above the thresholds
                if (edge_weight > max_weight and edge_weight != 1 and edge_weight > weight_threshold) or (
                    cosine_sim > max_cosine_sim and cosine_sim > cosine_threshold
                ):
                    max_weight = edge_weight
                    max_cosine_sim = cosine_sim
                    best_neighboring_community = communities[neighbor]["center"]
                    merged_by_metrics = True

                if merged_by_metrics == False:
                    delta_Q = self.calculate_delta_Q(node=node, target=neighbor, communities=communities, G=G)
                    print("cosine_sim: ", cosine_sim, " edge_weight: ", edge_weight, " cannot merge with these")
                    print("testing DQ: ", delta_Q, " with ", max_delta_Q)
                    max_delta_Q = delta_Q if delta_Q > max_delta_Q and delta_Q > 0 else max_delta_Q
                    best_neighboring_community = communities[neighbor]["center"]

            # if a good merge is found update the communities dict
            if best_neighboring_community:
                print("merging with community: ", best_neighboring_community)

                # add node to the target community [community_nodes]
                communities[best_neighboring_community]["community_nodes"].extend([node])

                # k_depth_neighbors of the community is now the community and the node k_depth_neighbors
                new_k_depth_neighbors = communities[best_neighboring_community]["k_depth_neighbors"]
                new_k_depth_neighbors.extend(communities[node]["k_depth_neighbors"])

                communities[best_neighboring_community]["k_depth_neighbors"] = list(set(new_k_depth_neighbors))

                # update the center of the node

                communities[node]["center"] = best_neighboring_community

            else:
                print("no good merges found")

            print("----------------")
        end = timer()
        print(end - start)

    def calculate_delta_Q(self, node: int, target: int, communities: dict, G: networkx.Graph):
        k_depth_subgraph_nodes = list(
            set(communities[node]["k_depth_neighbors"] + communities[target]["k_depth_neighbors"])
        )

        target_community_center = communities[target]["center"]
        target_community_nodes = communities[target_community_center]["community_nodes"]

        # size of all the links in the network (in our case the network is k_depth from node + k_depth from target)
        m = G.subgraph(k_depth_subgraph_nodes).size(weight="weight")

        k_i = G.degree(node, weight="weight")

        # sum of the weights from node i to nodes in C
        k_i_in = 0
        for target_community_neighbor in target_community_nodes:
            # handle networkx error if edges (node, target_community_neighbor) does not exist
            try:
                k_i_in += G[node][target_community_neighbor]["weight"]
            except Exception:
                pass

        # k_i_in = sum(
        #     G[node][target_community_neighbor]["weight"] for target_community_neighbor in target_community_nodes
        # )

        # sum of the weight in C
        S_in = G.subgraph(target_community_nodes).size(weight="weight")

        # sum of the weights of links incident to C
        S_tot = sum(element[1] for element in G.degree(target_community_nodes, weight="weight")) - S_in * 2

        delta_Q = (
            (S_in + k_i_in) / (2 * m)
            - (((S_tot + k_i) / (2 * m)) ** 2)
            - (S_in / (2 * m))
            + (S_tot / (2 * m)) ** 2
            + ((k_i / (2 * m)) ** 2)
        )

        return delta_Q  #! delta Q is too large?
