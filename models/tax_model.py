import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("Simplified_Tax_Levy_Prediction").getOrCreate()

    # load data
    df = spark.read.csv(in_directory, header=True, inferSchema=True).dropDuplicates()

    # filtering for certain colums
    # cast numerical columns to correct types
    numerical_columns = [
        "CURRENT_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_LAND_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "property_age",
    ]

    for col_name in numerical_columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # going to clean the data by transofrming and nulling 
    # log transformation for skewed features 
    for col_name in numerical_columns[:4]:  # only financial columns
        df = df.withColumn(f"log_{col_name}", log1p(col(col_name)))

    train_df = df.filter(col("TAX_LEVY") > 0).na.drop()

    # encode categorical columns since ml models dont work with nums
    categorical_cols = ["ZONING_DISTRICT", "NEIGHBOURHOOD_CODE"]
    for col_name in categorical_cols:
        # need to encode this cuz pyspark regress models dont like ints
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        train_df = indexer.fit(train_df).transform(train_df)

    feature_cols = (
        [f"log_{col_name}" for col_name in numerical_columns[:4]]  # log-transformed columns
        + numerical_columns[4:]  # property_age
        + [f"{col_name}_index" for col_name in categorical_cols]  # encoded categorical features
    )

    # assemble and scale some 
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    train_df = assembler.transform(train_df)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    train_df = scaler.fit(train_df).transform(train_df)


    spark.stop()


if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
