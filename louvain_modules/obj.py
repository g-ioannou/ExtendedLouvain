from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import subgraph
from pyspark.sql.types import ArrayType, FloatType, IntegerType
from pyspark.sql.window import Window
from louvain_modules.reader import Reader


from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame
from graphframes import GraphFrame
from pyspark.sql import functions as F

import numpy
import networkx


class Graph:
    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame, k_depth: int, spark: SparkSession) -> None:

        self.nodes = nodes_df
        self.edges = edges_df
        self.k_depth = k_depth
        self.spark = spark

    def modularity(self):

        sum_of_weights = self.edges.groupBy().agg(F.sum(F.col("weight")).alias("sum_of_graph_links"))

        sum_of_weights.show()

    def initialize_communities(self) -> DataFrame:

        self.nodes = (
            self.nodes.select("id", "vector")
            .withColumn("community_nodes", F.array("id"))
            .withColumn("community_edges", F.array())
            .withColumn("center", F.col("id"))
        )

    def to_nx(self) -> networkx.Graph:

        self._node_attr = self.nodes.drop("vector").toPandas().set_index("id").to_dict("index")

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
            .drop("id", "center", "community_nodes", "community_edges", "k_depth_neighbors", "depth_1_neighbors")
            .withColumnRenamed("vector", "vector_src")
            .join(self.nodes, on=self.nodes.id == self.edges.dst, how="inner")
            .drop("id", "center", "community_nodes", "community_edges", "k_depth_neighbors", "depth_1_neighbors")
            .withColumnRenamed("vector", "vector_dst")
            .withColumn("cosine", F.round(udf_numpy_cosine(F.col("vector_src"), F.col("vector_dst")), 3))
            .drop(*["vector_src", "vector_dst"])
        )

    def get_partition_using_metrics(self, weight_threshold: float, cosine_threshold: float, iteration: int) -> dict:
        # calculating S_in
        df_S_in = (
            self.nodes.withColumn("S_in", F.size("community_edges"))
            .filter(F.col("id") == F.col("center"))
            .select("center", "S_in")
        )

        # calculate S_tot
        temp_edges = (
            self.edges.join(
                self.nodes.select("id", "center"),
                on=self.edges.src == self.nodes.id,
                how="inner",
            )
            .drop("id")
            .withColumnRenamed("center", "center_src")
            .join(
                self.nodes.select("id", "center"),
                on=self.edges.dst == self.nodes.id,
                how="inner",
            )
            .withColumnRenamed("center", "center_dst")
            .drop("id")
        )

        temp_nodes = (
            self.nodes.join(temp_edges, on=((temp_edges.src == self.nodes.id) | (temp_edges.dst == self.nodes.id)))
            .filter(F.col("center_src") != F.col("center_dst"))
            .groupBy("center", "community_nodes", "community_edges", "k_depth_neighbors", "depth_1_neighbors", "vector")
            .agg(F.sum("weight").alias("S_tot"))  # ? S_tot
            .withColumn("S_in", F.size("community_edges"))  # ? S_in
            .drop("k_depth_neighbors", "depth_1_neighbors")
            .withColumnRenamed("center", "id")
            .checkpoint()
        )

        temp_edges_k_i_in = (
            temp_edges.filter(F.col("center_src") != F.col("center_dst"))
            .withColumn("src_dst_center", F.sort_array(F.array("center_src", "center_dst")))
            .groupBy("src_dst_center", "cosine")
            .agg(F.sum("weight").alias("weight"))
            .withColumn("src", F.col("src_dst_center")[0])
            .withColumn("dst", F.col("src_dst_center")[1])
            .drop("src_dst_center")
            .checkpoint()
        )
        # temp_nodes.show()
        temp_nodes = self.__find_k_depth_neighbors(temp_nodes, temp_edges_k_i_in, self.k_depth)

        temp_nodes.show(temp_nodes.count())
        # temp_edges_k_i_in.show()

        communities = set(temp_nodes.select("id").toPandas()["id"])
        return self.get_partition_using_metrics_v2(communities, weight_threshold, cosine_threshold, iteration)

    def get_partition_using_metrics_v2(
        self, communities: set, weight_threshold: float, cosine_threshold: float, iteration: int
    ):
        # dont't account weight for first iteration
        if iteration == 1:
            weight_threshold = float("inf")

        G = self.to_nx()

        # nodes dataframe collected as dict
        community_data = self._node_attr

        print(communities)
        print(community_data)

        # keeping track of merged communities so that double merges won't happen
        merged_communities = {}

        # communities contains community centers
        for community in communities:

            if community in merged_communities:
                continue

            print("checking community: ", community)
            community_nodes = community_data[community]["community_nodes"]
            community_k_depth_neighbors = set(community_data[community]["k_depth_neighbors"])

            # initialize best community to None
            best_neighboring_community = None

            max_cosine_sim = -1
            max_weight = -1
            max_delta_Q = -1

            checked_neighboring_communities = []

            for node in community_nodes:
                neighbors = G.neighbors(node)

                for neighbor in neighbors:
                    neighbor_community = community_data[neighbor]["center"]

                    # if a neighboring community is already checked procced to next neighbor, else append it  to chekced
                    if neighbor_community in checked_neighboring_communities or neighbor_community == community:
                        continue
                    else:
                        checked_neighboring_communities.append(neighbor_community)

                    # get node's community nodes and k_depth_neighbors
                    neighbor_community_nodes = community_data[neighbor_community]["community_nodes"]
                    neighbor_k_depth_neighbors = set(community_data[neighbor_community]["k_depth_neighbors"])

                    print(f"checking node {node} [in {community}] with {neighbor} [in {neighbor_community}]")
                    print(f"Community {community} k depth neighbors: {community_k_depth_neighbors}")
                    print(f"Community {neighbor_community} k depth neighbors: {neighbor_k_depth_neighbors}")
                    cosine_sim = G[node][neighbor]["cosine"]
                    edge_weight = self.calculate_edge_weight(
                        community_nodes,
                        community_k_depth_neighbors,
                        neighbor_community_nodes,
                        neighbor_k_depth_neighbors,
                    )

                    print("Cosine similarity:", cosine_sim)
                    print("Weight:", edge_weight)

                    # test merge first by cosine similarity
                    if cosine_sim > max_cosine_sim and cosine_sim > cosine_threshold:
                        max_cosine_sim = cosine_sim
                        max_weight = edge_weight
                        best_neighboring_community = neighbor_community
                    # then test merge by edge weight
                    elif cosine_sim == max_cosine_sim and edge_weight > max_weight and edge_weight > weight_threshold:
                        max_weight = edge_weight
                        best_neighboring_community = neighbor_community
                    # then test merge by DQ
                    elif edge_weight > max_weight and edge_weight > weight_threshold:
                        max_weight = edge_weight
                        best_neighboring_community = neighbor_community

            if best_neighboring_community != None:
                # change center for community (update all community nodes center)
                for node in community_nodes:
                    community_data[node]["center"] = best_neighboring_community

                # add community nodes to the new community (update the target community)
                community_data[best_neighboring_community]["community_nodes"].extend(community_nodes)

                # update k_depth_neighbors = UNION(k_depth_neighbors_src,k_depth_neighbors_dst)

                community_data[best_neighboring_community]["k_depth_neighbors"] = (
                    neighbor_k_depth_neighbors.union(community_k_depth_neighbors)
                    - set(neighbor_community_nodes)
                    - set(community_nodes)
                )

                community_data[best_neighboring_community]['community_edges'].extend(community_data[community]['community_edges'])

                merged_communities.update({community, best_neighboring_community})

            print(f"Best neighbor: {best_neighboring_community}")
            print(f"cosine similarity: {max_cosine_sim}")
            print(f"weight: {max_weight}")

            print("------------------")

        partition = {}

        for community in communities:
            community_nodes = community_data[community]["community_nodes"]
            for node in community_nodes:
                partition[node] = community

        return partition

    

    def calculate_edge_weight(
        self,
        src_community_nodes: list,
        src_k_depth_neighbors: list,
        dst_community_nodes: list,
        dst_k_depth_neighbors: list,
    ):

        src_community_nodes = set(src_community_nodes)
        dst_community_nodes = set(dst_community_nodes)
        src_k_depth_neighbors = set(src_k_depth_neighbors)
        dst_k_depth_neighbors = set(dst_k_depth_neighbors)

        # calculate INTERSECT(community_a_nodes , community_b_nodes) up to k-depth
        instersect__src_k_depth_neighbors__dst_community_nodes = src_k_depth_neighbors.intersection(dst_community_nodes)
        instersect__dst_k_depth_neighbors__src_community_nodes = dst_k_depth_neighbors.intersection(src_community_nodes)
        k_depth_intersect__src_community_nodes__dst_community_nodes = len(
            (
                instersect__dst_k_depth_neighbors__src_community_nodes.union(
                    instersect__src_k_depth_neighbors__dst_community_nodes
                )
            )
        )

        # calculate UNION(community_a_nodes, community_b_nodes)
        union__src_community_nodes__dst_community_nodes = len(src_community_nodes.union(dst_community_nodes))

        weight = (
            k_depth_intersect__src_community_nodes__dst_community_nodes
            / union__src_community_nodes__dst_community_nodes
        )

        return round(weight, 3)

    def __to_nx(self, nodes_df: DataFrame, edges_df: DataFrame) -> networkx.Graph:

        _node_attr = nodes_df.drop("vector").toPandas().set_index("id").to_dict("index")

        G_nx = networkx.from_pandas_edgelist(
            edges_df.toPandas(), source="src", target="dst", edge_attr=["weight", "cosine"]
        )

        networkx.set_node_attributes(G_nx, _node_attr)

        return G_nx

    def __find_k_depth_neighbors(self, nodes_df: DataFrame, edges_df: DataFrame, depth: int = 1):
        k_depth_neighbors = nodes_df.select("id").withColumn(f"depth_0_neighbor", F.col("id"))

        for i in range(1, depth + 1):
            k_depth_neighbors = k_depth_neighbors.withColumn(f"depth_{i}_neighbor", F.lit(None).cast(IntegerType()))

        for i in range(0, depth):
            k_depth_neighbors = (
                k_depth_neighbors.join(
                    edges_df.select("src", "dst"),
                    on=(
                        (edges_df.src == k_depth_neighbors[f"depth_{i}_neighbor"])
                        | (edges_df.dst == k_depth_neighbors[f"depth_{i}_neighbor"])
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

        nodes_df = (
            nodes_df.join(
                k_depth_neighbors,
                on=nodes_df.id == k_depth_neighbors.id_temp,
                how="inner",
            )
            .drop("id_temp")
            .select(  # ? THIS IS UNEEDED
                "id",
                "community_nodes",
                "community_edges",
                "k_depth_neighbors",
                "vector",
                "S_in",
                "S_tot",
            )
        )

        nodes_df = (
            nodes_df.join(
                edges_df.select("src", "dst"),
                on=((edges_df.src == nodes_df.id) | (edges_df.dst == nodes_df.id)),
                how="left",
            )
            .withColumn("depth_1_neighbors", F.when(F.col("id") == F.col("dst"), F.col("src")).otherwise(F.col("dst")))
            .drop("src", "dst")
            .groupBy("id", "community_nodes", "community_edges", "k_depth_neighbors", "vector", "S_in", "S_tot")
            .agg(F.collect_list("depth_1_neighbors").alias("depth_1_neighbors"))
        )

        return nodes_df

    def get_partition_using_metrics_old(self, weight_threshold: float, cosine_threshold: float, iteration):

        self.nodes = self.nodes.withColumn(
            "is_merged", F.when(F.col("id") != F.col("center"), F.lit(1)).otherwise(F.lit(0))
        )

        if iteration == 1:
            weight_threshold = float("inf")

        G = self.to_nx()
        self.nodes = self.nodes.drop("is_merged")

        communities = self._node_attr

        for node in communities:

            if communities[node]["is_merged"] == 1:

                continue
            else:
                print("checking node: ", node)

            neighbors = communities[node]["depth_1_neighbors"]

            # find all neighbors of node i
            node_neighbors = {
                neighbor: {
                    "center": communities[neighbor]["center"],
                    "community_nodes": communities[neighbor]["community_nodes"],
                }
                for neighbor in neighbors
            }

            # initialize best community to None
            best_neighboring_community = None
            max_cosine_sim = -1
            max_weight = -1
            max_delta_Q = -1

            merged_by_metrics = False
            for neighbor in node_neighbors:
                if communities[neighbor]["center"] == communities[node]["center"]:
                    continue
                try:
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
                        delta_Q = self.calculate_delta_Q(
                            node=node, target=neighbor, communities=communities, G=G, iteration=iteration
                        )
                        # print("cosine_sim: ", cosine_sim, " edge_weight: ", edge_weight, " cannot merge with these")
                        # print("testing DQ: ", delta_Q, " with ", max_delta_Q)
                        max_delta_Q = delta_Q if delta_Q > max_delta_Q and delta_Q > 0 else max_delta_Q
                        best_neighboring_community = communities[neighbor]["center"]
                except Exception:
                    pass

            # if a good merge is found update the communities dict
            if best_neighboring_community:

                print("merging ", node, " with community: ", best_neighboring_community)

                # add node to the target community [community_nodes]
                communities[best_neighboring_community]["community_nodes"].extend(communities[node]["community_nodes"])

                # k_depth_neighbors of the community is now the community and the node k_depth_neighbors
                new_k_depth_neighbors = communities[best_neighboring_community]["k_depth_neighbors"]
                new_k_depth_neighbors.extend(communities[node]["k_depth_neighbors"])

                communities[best_neighboring_community]["k_depth_neighbors"] = list(set(new_k_depth_neighbors))

                # update the center of the node
                for community_node in communities[node]["community_nodes"]:
                    communities[community_node]["center"] = best_neighboring_community

                communities[node]["is_merged"] = 1
                communities[best_neighboring_community]["is_merged"] = 1
            else:
                ...
                # print("no good merges found")

            # print("----------------")

        partition = {node: communities[node]["center"] for node in communities}

        return partition

    def find_k_depth_neighbors(self) -> DataFrame:

        depth = self.k_depth

        k_depth_neighbors = (
            self.nodes.select("id").withColumn(f"depth_0_neighbor", F.col("id")).filter(F.col("center") == F.col("id"))
        )

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
                "vector",
            )
        )

        self.nodes = (
            self.nodes.join(
                self.edges.select("src", "dst"),
                on=((self.edges.src == self.nodes.id) | (self.edges.dst == self.nodes.id)),
                how="left",
            )
            .withColumn("depth_1_neighbors", F.when(F.col("id") == F.col("dst"), F.col("src")).otherwise(F.col("dst")))
            .drop("src", "dst")
            .groupBy("id", "center", "community_nodes", "community_edges", "k_depth_neighbors", "vector")
            .agg(F.collect_list("depth_1_neighbors").alias("depth_1_neighbors"))
        )

    def calculate_delta_Q(self, node: int, target: int, communities: dict, G: networkx.Graph, iteration: int):

        k_depth_neighbors_combined = list(
            set(communities[node]["k_depth_neighbors"] + communities[target]["k_depth_neighbors"])
        )

        k_depth_subgraph_nodes = []

        for node in k_depth_neighbors_combined:
            k_depth_subgraph_nodes.extend(communities[node]["community_nodes"])

        k_depth_subgraph_nodes.extend(communities[node]["community_nodes"])
        k_depth_subgraph_nodes.extend(communities[target]["community_nodes"])

        k_depth_subgraph_nodes = list(set(k_depth_subgraph_nodes))

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

        # sum of the weights in C
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

        return delta_Q

    def update_with_partition(self, partition: dict):
        partition_zipped = list(zip(partition.keys(), partition.values()))

        new_partition_df = (
            self.spark.createDataFrame(partition_zipped)
            .withColumnRenamed("_1", "id_temp")
            .withColumnRenamed("_2", "new_community")
        )

        self.nodes = (
            self.nodes.join(new_partition_df, on=new_partition_df.id_temp == self.nodes.id, how="inner")
            .drop("center", "id_temp")
            .withColumnRenamed("new_community", "center")
            .groupBy("center")
            .agg(
                F.collect_set("id").alias("community_nodes"),
                F.collect_list("k_depth_neighbors").alias("k_depth_neighbors"),
                F.collect_list("vector").alias("__vectors"),
                F.collect_set("depth_1_neighbors").alias("depth_1_neighbors"),
            )
            .withColumn("k_depth_neighbors", F.flatten("k_depth_neighbors"))
            .withColumn("k_depth_neighbors", F.array_distinct("k_depth_neighbors"))
            .withColumn("k_depth_neighbors", F.array_except("k_depth_neighbors", "community_nodes"))
            .withColumn(
                "depth_1_neighbors", F.array_except(F.array_distinct(F.flatten("depth_1_neighbors")), "community_nodes")
            )
            .withColumn("id", F.explode("community_nodes"))
        )

        self.edges = (
            self.edges.drop("cosine", "weight")
            .join(self.nodes, on=self.nodes.id == self.edges.src)
            .drop("id", "vectors", "__vectors", "depth_1_neighbors")
            .withColumnRenamed("center", "src_community")
            .withColumnRenamed("k_depth_neighbors", "src_k_depth_neighbors")
            .withColumnRenamed("community_nodes", "src_community_nodes")
            .join(self.nodes, on=self.nodes.id == self.edges.dst)
            .drop("id", "vectors", "__vectors", "depth_1_negihbors")
            .withColumnRenamed("center", "dst_community")
            .withColumnRenamed("community_nodes", "dst_community_nodes")
            .withColumnRenamed("k_depth_neighbors", "dst_k_depth_neighbors")
            .withColumn(
                "intersect_(src_k_depth_neighbors, dst_community_nodes)",
                F.array_intersect("src_k_depth_neighbors", "dst_community_nodes"),
            )
            .withColumn(
                "intersect_(dst_k_depth_neighbors, src_community_nodes)",
                F.array_intersect("dst_k_depth_neighbors", "src_community_nodes"),
            )
            .withColumn(
                "k_depth_intersect_(src_community_nodes,dst_community_nodes)",
                F.array_union(
                    "intersect_(src_k_depth_neighbors, dst_community_nodes)",
                    "intersect_(dst_k_depth_neighbors, src_community_nodes)",
                ),
            )
            .withColumn(
                "union_(src_community_nodes,dst_community_nodes)",
                F.array_union("src_community_nodes", "dst_community_nodes"),
            )
            .withColumn(
                "weight",
                F.when(F.col("src_community") == F.col("dst_community"), 1).otherwise(
                    F.round(
                        F.size("k_depth_intersect_(src_community_nodes,dst_community_nodes)")
                        / F.size("union_(src_community_nodes,dst_community_nodes)"),
                        scale=3,
                    )
                ),
            )
            .select("src", "dst", "weight", "src_community", "dst_community")
            .checkpoint()
        )
        community_edges = (
            self.edges.filter(F.col("src_community") == F.col("dst_community"))
            .withColumn("src_dst", F.array("src", "dst"))
            .groupBy("src_community", "dst_community")
            .agg(F.collect_list("src_dst").alias("community_edges"))
            .drop("dst_community")
            .withColumnRenamed("src_community", "center_tmp")
        )

        @F.udf(ArrayType(FloatType()))
        def udf_median(vectors: list):

            return numpy.median(vectors, axis=0).tolist()

        # self.nodes = (
        #     self.nodes.withColumn("vector", udf_median("__vectors")).drop("__vectors").join(community_edges, ["center"],'left')
        # )

        self.nodes = (
            self.nodes.withColumn("vector", udf_median("__vectors"))
            .drop("__vectors")
            .join(community_edges, on=community_edges.center_tmp == self.nodes.center, how="left")
        ).drop("center_tmp")

        self.edges = self.edges.drop("src_community", "dst_community")
