import json

from pyspark.sql.session import SparkSession


class Reader:
    @staticmethod
    def read_json(path: str) -> dict:
        with open(path, "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def read_spark_csv(spark:SparkSession, path, delimiter, header):
        return spark.read.option("delimiter",delimiter).csv(path,header=header,inferSchema=True)

    def read_spark_parquet():
        ...
