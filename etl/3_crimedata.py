from pyspark.sql import SparkSession

# Initialize Spark session with Excel support
spark = SparkSession.builder.appName("Read Excel File").config("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.13.5").getOrCreate()

# Read the Excel file
data = spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").option("dataAddress", "'Sheet1'!A1").load("data/crimedata.xls")

data.write.csv("data_crime", header=True, mode="overwrite", compression="gzip")
#data.show()
