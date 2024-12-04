"""
This script filters the 1GB dataset based on the features needed for further analysis using spark.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, when

spark = SparkSession.builder.appName("Real Estate Analysis").getOrCreate()

# Load the dataset
file_path = "data/california_prices.csv" 
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Filter fbased on price not NULL
filtered_data = data.filter(
    (col("sale_price").isNotNull())
)

# Year and Month
enriched_data = filtered_data.withColumn("sale_year", year(col("sale_datetime"))) \
                             .withColumn("sale_month", month(col("sale_datetime")))

# Select relevant columns
selected_data = enriched_data.select(
    col("sale_price"),
    col("property_type"),
    col("property_city"),
    col("property_county"),
    col("sale_year"),
    col("sale_month"),    
)

selected_data.show(10)

output_path = "data_partitioned"
selected_data.write.csv(output_path, header=True, mode="overwrite", compression="gzip")
