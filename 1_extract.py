from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, when

spark = SparkSession.builder.appName("Data").getOrCreate()

# Load the dataset
file_path = "data/property-tax-report.csv" 
data = spark.read.csv(file_path, header=True, inferSchema=True)

output_path = "data_partitioned"
data.write.csv(output_path, header=True, mode="overwrite", compression="gzip")