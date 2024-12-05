from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim, regexp_replace, when

spark = SparkSession.builder.appName("Join Weather and Filtered Data").getOrCreate()

# Read the weather data
weather_data = spark.read.csv("data/weather.csv", header=True, inferSchema=True)

# Read the filtered residential data
data_filtered = spark.read.csv("data_filtered", header=True, inferSchema=True)

# Normalize the "Name" column in weather data to remove "County" and make it comparable
weather_data = weather_data.withColumn(
    "normalized_name",
    upper(trim(regexp_replace(col("Name"), " County", "")))
)

# Normalize the "property_city" column in data_filtered for comparison
data_filtered = data_filtered.withColumn(
    "normalized_property_city",
    upper(trim(col("property_city")))
)

# Join the datasets on the normalized columns
joined_data = data_filtered.join(
    weather_data.select("normalized_name", "Value"),
    data_filtered["normalized_property_city"] == weather_data["normalized_name"],
    "left"
)

# Rename the "Value" column to "temperature"
joined_data = joined_data.withColumnRenamed("Value", "temperature")

# Drop rows with no temperature data
joined_data = joined_data.filter(col("temperature").isNotNull())

# Drop intermediate columns used for normalization
joined_data = joined_data.drop("normalized_property_city", "normalized_name")

# Show the resulting data
joined_data.show()

# Write the final dataset to a CSV file
joined_data.write.csv("data_clean", header=True, mode="overwrite", compression="gzip")
