from louvain_modules.obj import Graph
from louvain_modules.reader import Reader

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from louvain_modules.spark_tools import calculate_dummy_vectors
from pyspark.sql import functions as F

import argparse
import os
import shutil


class Initializer:
    def __init__(self) -> None:
        self.protject_path = Path(__file__).parent.parent

        arg_parser = argparse.ArgumentParser(
            description="Distributed implementation of Louvain's community detection algorithm, using user profile information."
        )

        arg_parser.add_argument(
            "-c",
            "--config",
            metavar="",
            required=True,
            help='Configuration file name (ex. "config.json"). Config file must be inside the /configs folder, located in this project\'s main directory',
        )

        self.config = Reader.read_json(f"{self.protject_path}/configs/{arg_parser.parse_args().config}")

        self.parameters = self.config["parameters"]
        self.k_depth = self.parameters["k_depth"]

        self.make_dirs()

    def initialize_graph(self):
        feature_cols = self.config["graph"]["nodes"]["features"]
        new_feature_cols = []

        nodes_df = Reader.read_spark_csv(
            spark=self.spark,
            path=self.config["graph"]["nodes"]["file_path"],
            delimiter=self.config["graph"]["nodes"]["delimiter"],
            header=self.config["graph"]["nodes"]["has_header"],
        ).withColumnRenamed(self.config["graph"]["nodes"]["id_column"], "id_cleaned")

        cols_to_keep = ["id_cleaned"]
        feature_counter = 1
        for feature_col in feature_cols:
            nodes_df = nodes_df.withColumnRenamed(feature_col, f"feature_{feature_counter}")
            cols_to_keep.append(f"feature_{feature_counter}")
            new_feature_cols.append(f"feature_{feature_counter}")
            feature_counter += 1

        nodes_df = nodes_df.select(cols_to_keep).withColumnRenamed("id_cleaned", "id")

        nodes_df = calculate_dummy_vectors(nodes_df).select("id", "vector")
        nodes_df = nodes_df.withColumn("vector", nodes_df.vector.cast("array<float>"))

        edges_df = (
            Reader.read_spark_csv(
                spark=self.spark,
                path=self.config["graph"]["edges"]["file_path"],
                delimiter=self.config["graph"]["edges"]["delimiter"],
                header=self.config["graph"]["edges"]["has_header"],
            )
            .withColumnRenamed(self.config["graph"]["edges"]["source_col_name"], "src")
            .withColumnRenamed(self.config["graph"]["edges"]["target_col_name"], "dst")
            .withColumn('weight',F.lit(1))
        )

        
        G = Graph(nodes_df, edges_df, self.spark)
        
        
        G.filter_out_small_communities()
        

        return G

    def initialize_spark(self) -> SparkSession:
        try:
            spark_conf = self.__configure_spark()
            self.spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
            self.spark.sparkContext.setCheckpointDir("checkpoints")
            self.spark.sparkContext.addPyFile("packages/enchant.zip")
            self.spark.sparkContext.addPyFile("packages/graphframes-8a81b8128f6fc4890b52ffea63e8442b923dc962.zip")
            return self.spark
        except Exception as e:
            print("Error while initializing Spark.")
            print(e)

    def set_motif(self, depth: int) -> str:

        motif = "(n0)-[e0]->"

        for i in range(1, depth):
            motif += f"(n{i});(n{i})-[e{i}]->"

        motif += f"(nK)"

        return motif

    def make_dirs(self):

        try:
            shutil.rmtree(f"{self.protject_path}/data/temp")
        except Exception:
            pass

        os.mkdir(f"{self.protject_path}/data/temp")

        try:
            shutil.rmtree(f"{self.protject_path}/checkpoints")
        except Exception:
            pass

        os.mkdir(f"{self.protject_path}/checkpoints")

        try:
            shutil.rmtree(f"{self.protject_path}/spark-warehouse")
        except Exception:
            pass

        os.mkdir(f"{self.protject_path}/spark-warehouse")

    def __configure_spark(self) -> SparkConf:
        spark_conf = SparkConf()
        [spark_conf.set(str(key), str(value)) for key, value in self.config["spark"].items()]
        return spark_conf
