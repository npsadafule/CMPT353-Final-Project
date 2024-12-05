from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim, regexp_replace

# Initialize Spark session
spark = SparkSession.builder.appName("Join Crime and Residential Data").getOrCreate()

# Read the crime data
data_crime = spark.read.csv("data_crime", header=True, inferSchema=True)

# Read the residential data (CSV file)
data_residential = spark.read.csv("data_residential", header=True, inferSchema=True)

# Normalize city names in both datasets
data_crime = data_crime.withColumn(
    "normalized_city",
    upper(trim(regexp_replace(col("City"), "[^A-Za-z0-9 ]", "")))
)

data_residential = data_residential.withColumn(
    "normalized_property_city",
    upper(trim(regexp_replace(col("property_city"), "[^A-Za-z0-9 ]", "")))
)

# Join datasets on the normalized city names
joined_data = data_residential.join(
    data_crime,
    data_residential["normalized_property_city"] == data_crime["normalized_city"],
    "left"
)

# Drop intermediate
final_data = joined_data.drop("normalized_city", "normalized_property_city", "City")

# Write the result to a CSV file
final_data.write.csv("data_joined", header=True, mode="overwrite", compression="gzip")

final_data.show()
