import networkx
from pyspark.sql.dataframe import DataFrame
from pyvis.network import Network
from louvain_modules.obj import Graph

import random
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType


class Visualizer:
    def visualize(self, G: Graph, iteration: int = -1):
        nt = Network("95%", "100%", heading="Graph")

        viz_nodes = G.nodes

        node_count = viz_nodes.count()

        map_size = 15 * node_count

        @F.udf(IntegerType())
        def udf_random_position():
            return random.randint(0, map_size)

        @F.udf(StringType())
        def udf_random_color():
            random_number = random.randint(0, 16777215)
            hex_number = str(hex(random_number))
            hex_number = "#" + hex_number[2:]
            return hex_number

        viz_nodes = (
            viz_nodes.groupby("center")
            .agg(F.collect_list("id").alias("id"))
            .withColumn("x", udf_random_position())
            .withColumn("color", udf_random_color())
            .withColumn("id", F.explode("id"))
        )
        viz_nodes= (
            viz_nodes.groupby("center", "x", "color")
            .agg(F.collect_list("id").alias("id"))
            .withColumn("y", udf_random_position())
            .withColumn("id", F.explode("id"))
            .withColumn(
                "title", F.concat(F.lit("ID: "), F.col("id"), F.lit("<br>Community: "), F.col("center").cast("string"))
            )
        )

        nx_graph = self._nx_graph(viz_nodes, G.edges)

        nt.set_options(
            """
            var options = {
                "configure":{
                    "enabled":true
                },
                "edges":{
                    "smooth":false
                },
                "physics": {
                    "barnesHut": {
                    "gravitationalConstant": -150,
                    "centralGravity": 0,
                    "springConstant": 0,
                    "damping": 1,
                    "avoidOverlap": 1
                    },
                    "maxVelocity": 1,
                    "minVelocity": 1
                }  
            }"""
        )

        nt.from_nx(nx_graph)

        if iteration == -1:
            name = "graph.html"
        else:
            name = f"iteration_{iteration}_graph.html"

        nt.save_graph(name)

    def _nx_graph(self, nodes: DataFrame, edges: DataFrame):
        node_attrs = nodes.toPandas().set_index("id").to_dict("index")
        edges = edges.withColumn(
            "title",
            F.concat(
                F.lit("Source: "),
                F.col("src"),
                F.lit("<br>Target: "),
                F.col("dst"),
                F.lit("<br>Weight: "),
                F.col("weight"),
                F.lit("<br>Cosine similiratiy: "),
                F.col("cosine"),
            ),
        )
        nx_graph = networkx.from_pandas_edgelist(edges.toPandas(), source="src", target="dst", edge_attr=["title"])
        networkx.set_node_attributes(nx_graph, node_attrs)
        return nx_graph
