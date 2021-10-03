from pyspark.sql.dataframe import DataFrame

from pyspark.ml.feature import StopWordsRemover, StringIndexer
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

import enchant

english_dict = enchant.Dict("en_US")


def calculate_dummy_vectors(nodes_df: DataFrame):
    @F.udf(ArrayType(StringType()))
    def udf_check_words(sentence: list):
        cleaned_words = []
        try:
            for word in sentence:
                if english_dict.check(word):
                    cleaned_words.append(word.lower())
        except Exception:
            pass

        if len(cleaned_words) == 0:
            cleaned_words.append("other")

        return list(set(cleaned_words))

    stop_word_list = StopWordsRemover().getStopWords()
    stop_word_list.remove("other")

    removers = []

    for col in nodes_df.columns:
        # do some basic cleaning using regex and tokenize the feature
        if "feature_" in col:
            nodes_df = nodes_df.withColumn(
                col,
                F.array_distinct(
                    F.array_remove(F.split(F.regexp_replace((F.col(col)), r"[^a-zA-Z ]", ""), "\s"), "")
                ),
            ).withColumn(col, udf_check_words(col))

            removers.append(StopWordsRemover(inputCol=col, outputCol=f"{col}_cleaned", stopWords=stop_word_list))

    # stop_word_list = nltk.corpus.stopwords.words("english")
    pipeline = Pipeline(stages=removers).fit(nodes_df)
    nodes_df = pipeline.transform(nodes_df)

    nodes_df = nodes_df.select("id", *[col for col in nodes_df.columns if "_cleaned" in col])

    for col in nodes_df.columns:
        if "_cleaned" in col:
            nodes_df = nodes_df.withColumnRenamed(col, col.replace("_cleaned", ""))

    nodes_df = map_lesser_features_to_other(nodes_df)
    
    nodes_df = vectorize_features(nodes_df)
    
    return nodes_df


def vectorize_features(nodes_df: DataFrame):
    nodes_df = nodes_df.withColumn("vector", F.array())

    for col in nodes_df.columns:
        if "feature_" in col:
            nodes_df = nodes_df.withColumn("vector", F.concat(F.col("vector"), F.col(col)))
    
    vector_values = nodes_df.withColumn("vector_exploded", F.explode("vector")).groupBy("vector_exploded").count().drop(
        "count"
    ).withColumn("strength", F.when(F.col("vector_exploded") == F.lit("other"), 1).otherwise(F.lit(100))).rdd.map(lambda row: (row[0],row[1])).collectAsMap()
    
    @F.udf(ArrayType(StringType()))
    def udf_calculate_vector(vector_array:list):
        vector = []

        for word in vector_values:
            if word in vector_array:
                vector.append(float(vector_values[word]))
            else:
                vector.append(float(0))
    
        return vector
    
    nodes_df = nodes_df.withColumn('vector', udf_calculate_vector('vector'))
    
    return nodes_df

def map_lesser_features_to_other(nodes_df: DataFrame):
    final_words_dict = {}
    for col in nodes_df.columns:
        if "feature_" in col:
            words_df = (
                nodes_df.select(col)
                .withColumn("word", F.explode(col))
                .groupBy("word")
                .count()
                .sort("count", ascending=False)
            )
            total_words = words_df.select(F.sum("count")).collect()[0][0]

            words_df = words_df.withColumn("percentage", F.round(F.col("count") / F.lit(total_words), 2)).withColumn(
                "new_word", F.when(F.col("percentage") < F.lit(0.01), "other").otherwise(F.col("word"))
            )

            words_mapping = words_df.select("word", "new_word").rdd.map(lambda row: (row[0], row[1])).collectAsMap()
            words_mapping = {col: words_mapping}
            final_words_dict.update(words_mapping)

    @F.udf(ArrayType(StringType()), StringType())
    def udf_map_to_other(word_array: list, col: str):
        new_word_list = []

        for word in word_array:
            new_word_list.append(final_words_dict[col][word])

        return new_word_list

    for col in nodes_df.columns:
        if "feature_" in col:
            nodes_df = nodes_df.withColumn(col, F.array_distinct(udf_map_to_other(col, F.lit(col))))

    return nodes_df
