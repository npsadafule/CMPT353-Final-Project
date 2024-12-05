from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark session
spark = SparkSession.builder.appName("Check Matching Crime Data").getOrCreate()

# Read the datasets
joined_data = spark.read.csv("data_joined", header=True, inferSchema=True)

# Add a column to check if crime data exists
joined_data = joined_data.withColumn(
    "has_crime_data",
    when(col("Violent crime").isNotNull(), True).otherwise(False)
)

# Count rows without matching crime data
unmatched_count = joined_data.filter(~col("has_crime_data")).count()
total_count = joined_data.count()

# Print summary results
print(f"Total entries in data_residential: {total_count}")
print(f"Entries without matching crime data: {unmatched_count}")

# Filter out rows without matching crime data
filtered_data = joined_data.filter(col("has_crime_data"))

# Drop the helper column 'has_crime_data'
filtered_data = filtered_data.drop("has_crime_data")

filtered_data.show()

# Write the filtered data to a new CSV file
filtered_data.write.csv("data_filtered", header=True, mode="overwrite", compression="gzip")
