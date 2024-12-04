"""Script to find the unique values in a column of a DataFrame."""

from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Unique Values in Column") \
    .getOrCreate()

# Load the dataset
file_path = "data_partitioned"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Specify the column name
column_name = "property_type"

# Select distinct values from the column
unique_values = data.select(column_name).distinct()

# Show unique values
unique_values.show(truncate=False)

# Optionally collect unique values into a Python list
unique_values_list = unique_values.rdd.map(lambda row: row[0]).collect()
print("Unique Values:", unique_values_list)
