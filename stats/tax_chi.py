import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, count
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
from scipy.stats import chi2_contingency
import pandas as pd

def main(in_directory):
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Chi-Square Test Analysis").getOrCreate()

    # Load data
    data = spark.read.csv(in_directory, header=True, inferSchema=True)

    # Filter rows where necessary columns are not null
    data = data.filter(col("CURRENT_LAND_VALUE").isNotNull() & col("TAX_LEVY").isNotNull())

    # Categorize CURRENT_LAND_VALUE into bins
    data = data.withColumn(
        "LAND_VALUE_CATEGORY",
        when(col("CURRENT_LAND_VALUE") < 1_000_000, "Low")
        .when((col("CURRENT_LAND_VALUE") >= 1_000_000) & (col("CURRENT_LAND_VALUE") < 2_000_000), "Medium")
        .otherwise("High")
    )

    # Function to perform Chi-Square Test
    def perform_chi2_test(df, row_var, col_var):
        contingency_table = (
            df.groupBy(row_var, col_var)
            .agg(count("*").alias("count"))
            .groupBy(row_var)
            .pivot(col_var)
            .sum("count")
            .fillna(0)
        )

        contingency_table_pd = contingency_table.toPandas().set_index(row_var)

        # Perform Chi-Square Test
        chi2, p, dof, expected = chi2_contingency(contingency_table_pd)
        print(f"Chi-Square Test Results for {row_var} vs {col_var}")
        print(f"Chi2 Statistic: {chi2}")
        print(f"P-Value: {p}")
        print(f"Degrees of Freedom: {dof}")
        print("Expected Frequencies:")
        print(pd.DataFrame(expected, columns=contingency_table_pd.columns, index=contingency_table_pd.index))
        print()

    # Chi-Square Test 1: LAND_VALUE_CATEGORY vs TAX_LEVY category
    data = data.withColumn(
        "TAX_CATEGORY",
        when(col("TAX_LEVY") < 1_000, "Low")
        .when((col("TAX_LEVY") >= 1_000) & (col("TAX_LEVY") < 10_000), "Medium")
        .otherwise("High")
    )

    print("Performing Chi-Square Test between LAND_VALUE_CATEGORY and TAX_CATEGORY...")
    perform_chi2_test(data, "LAND_VALUE_CATEGORY", "TAX_CATEGORY")

    # Chi-Square Test 2: ZONING_CLASSIFICATION vs TAX_CATEGORY
    print("Performing Chi-Square Test between ZONING_CLASSIFICATION and TAX_CATEGORY...")
    perform_chi2_test(data, "ZONING_CLASSIFICATION", "TAX_CATEGORY")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    in_directory = sys.argv[1]
    main(in_directory)
