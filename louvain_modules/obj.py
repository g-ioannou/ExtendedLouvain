from pyspark.sql.types import ArrayType, FloatType, IntegerType
from louvain_modules.reader import Reader


from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame
from graphframes import GraphFrame
from pyspark.sql import functions as F
import numpy

import networkx as nx


class Graph:
    def __init__(self, nodes_df: DataFrame, edges_df: DataFrame, spark: SparkSession) -> None:

        self.nodes = nodes_df
        self.edges = edges_df

    def modularity(self):
        
        sum_of_weights = self.edges.groupBy().agg(F.sum(F.col('weight')).alias('sum_of_graph_links'))

        sum_of_weights.show()

    def communities(self) :
        self.communities_df = (
            self.nodes.select("id", "vector")
            .withColumn("community_nodes", F.array("id"))
            .withColumn("community_edges", F.array())
            .withColumn("community_vectors", F.array("vector"))
            .withColumn("S_in", F.lit(0))
            .join(
                self.edges,
                on=((self.nodes["id"] == self.edges['src']) | (self.nodes['id'] == self.edges['dst'])),
                how="inner",
            )
            .withColumn("dst", F.when(F.col("id") == F.col("src"), F.col("dst")).otherwise(F.col("src")))
            .drop("src")
            .groupBy("id", "community_nodes", "community_edges", "S_in", "community_vectors")
            .agg(F.count(F.col('dst')).alias("S_tot"))
            .select("id", "community_nodes", "community_edges", "S_in", "S_tot", "community_vectors")
        )
        
        return self.communities_df

    def to_nx(self):
        return nx.from_pandas_edgelist(self.edges.toPandas(), source="src", target="dst")

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
            return float(inner_product / norms_product)

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

    def find_k_depth_neighbors(self, motif: str):
        g = GraphFrame(self.communities_df, self.edges)

        motifs = g.find(motif).filter(F.col("n0") != F.col("nK"))

        # motifs = motifs.withColumn('test', F.struct(F.col('a'),F.col('z')))

        return motifs
