from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
from pyspark.sql.functions import when, col
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
import sys

#use data partioned!
# Define schema
schema = StructType([
    StructField("PID", StringType(), True),
    StructField("LEGAL_TYPE", StringType(), True),
    StructField("FOLIO", StringType(), True),
    StructField("LAND_COORDINATE", StringType(), True),
    StructField("ZONING_DISTRICT", StringType(), True),
    StructField("ZONING_CLASSIFICATION", StringType(), True),
    StructField("LOT", StringType(), True),
    StructField("PLAN", StringType(), True),
    StructField("BLOCK", StringType(), True),
    StructField("DISTRICT_LOT", StringType(), True),
    StructField("FROM_CIVIC_NUMBER", StringType(), True),
    StructField("TO_CIVIC_NUMBER", StringType(), True),
    StructField("STREET_NAME", StringType(), True),
    StructField("PROPERTY_POSTAL_CODE", StringType(), True),
    StructField("NARRATIVE_LEGAL_LINE1", StringType(), True),
    StructField("NARRATIVE_LEGAL_LINE2", StringType(), True),
    StructField("NARRATIVE_LEGAL_LINE3", StringType(), True),
    StructField("NARRATIVE_LEGAL_LINE4", StringType(), True),
    StructField("NARRATIVE_LEGAL_LINE5", StringType(), True),
    StructField("CURRENT_LAND_VALUE", FloatType(), True),
    StructField("CURRENT_IMPROVEMENT_VALUE", FloatType(), True),
    StructField("TAX_ASSESSMENT_YEAR", StringType(), True),
    StructField("PREVIOUS_LAND_VALUE", FloatType(), True),
    StructField("PREVIOUS_IMPROVEMENT_VALUE", FloatType(), True),
    StructField("YEAR_BUILT", StringType(), True),
    StructField("BIG_IMPROVEMENT_YEAR", StringType(), True),
    StructField("TAX_LEVY", DoubleType(), True),
    StructField("NEIGHBOURHOOD_CODE", StringType(), True),
    StructField("REPORT_YEAR", StringType(), True)
])

def main(in_directory):
    
    spark = SparkSession.builder.appName("Chi-Square Test on Multiple Variables").getOrCreate()

    #  reading the data with a semicolon delimter
    data = spark.read.csv(in_directory, schema=schema, header=True, sep=";")

    # creates a new column for total house value which is created using the current value and current improvement
    data = data.withColumn(
        "TOTAL_HOUSE_VALUE",
        (col("CURRENT_LAND_VALUE") + col("CURRENT_IMPROVEMENT_VALUE"))
    )

    # filters the rows where TOTAL_HOUSE_VALUE is null
    filtered_data = data.filter(col("TOTAL_HOUSE_VALUE").isNotNull())

    # categorize house prices
    filtered_data = filtered_data.withColumn(
        "PRICE_CATEGORY",
        when(col("TOTAL_HOUSE_VALUE") < 1_000_000, "Less than 1M")
        .when((col("TOTAL_HOUSE_VALUE") >= 1_000_000) & (col("TOTAL_HOUSE_VALUE") <= 2_000_000), "1M-2M")
        .otherwise("More than 2M")
    )

    #  chi-square test function: ZONING_DISTRICT vs PRICE_CATEGORY
    def perform_chi2_test(df, row_var, col_var):
        contingency_table = df.groupBy(row_var, col_var) \
                              .count() \
                              .groupBy(row_var) \
                              .pivot(col_var) \
                              .sum("count") \
                              .fillna(0)
        contingency_table_pd = contingency_table.toPandas().set_index(row_var)
        contingency_table_pd = contingency_table_pd.loc[
            ~(contingency_table_pd < 5).any(axis=1)
        ]
        contingency_table_pd = contingency_table_pd.loc[
            :, ~(contingency_table_pd < 5).any(axis=0)
        ]
        if contingency_table_pd.shape[0] == 0 or contingency_table_pd.shape[1] == 0:
            print(f"Not enough data for a valid chi-square test for {row_var} and {col_var}.")
            return
        chi2, p, dof, expected = chi2_contingency(contingency_table_pd)
        print(f"Chi-Square Test Results for {row_var} vs {col_var}")
        print(f"Chi2 Statistic: {chi2}")
        print(f"P-Value: {p}")
        print(f"Degrees of Freedom: {dof}")
        print("Expected Frequencies:")
        print(expected)
        print()

    #  chi-Square test1: ZONING_DISTRICT vs PRICE_CATEGORY
    perform_chi2_test(filtered_data, "ZONING_DISTRICT", "PRICE_CATEGORY")

    # chi-Square test2: YEAR_BUILT vs PRICE_CATEGORY
    perform_chi2_test(filtered_data, "YEAR_BUILT", "PRICE_CATEGORY")

if __name__ == '__main__':
    in_directory = sys.argv[1]
    main(in_directory)
