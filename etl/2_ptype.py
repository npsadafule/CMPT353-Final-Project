"""
This script filters the the dataset by only selecting residential property types.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month

# Initialize Spark session
spark = SparkSession.builder.appName("Real Estate Analysis").getOrCreate()

# Load the dataset
file_path = "data_partitioned" 
data = spark.read.csv(file_path, header=True, inferSchema=True)

# List of Residential Property Types
residential_types = [
    'SINGLE FAMILY RESIDENCE', 'SINGLE FAMILY DWELLING', 
    'MA-SINGLE FAMILY DWELLING', 'PI-SINGLE FAMILY DWELLING',
    'MULTI-FAMILY RESIDENCE', 'FOURPLEX', 'DUPLEX', 'TRIPLEX',
    'APARTMENT 5 - 10 UNITS', 'APARTMENT 11 - 20 UNITS', 
    'APARTMENT 21 - 40 UNITS', 'APARTMENT 41 - 60 UNITS', 
    'APARTMENT 60 - 100 UNITS', 'APARTMENT OVER 100 UNITS',
    'RESIDENTIAL CONDOMINIUM', 'MA-RESIDENTIAL CONDOMINIUM', 'CONDOMINIUM',
    'TOWNHOUSE', 
    'RESIDENTIAL LAND W/MISC IMPS < 1 ACRE',
    'RESIDENTIAL LAND W/MISC IMP 1-4.9 ACRES', 
    'RESIDENTIAL LAND W/MISC IMP 5-9.9 ACRES', 
    'RESIDENTIAL LAND W/MISC IMP 10-49.9 ACRES', 
    'RESIDENTIAL LAND W/MISC IMP 50-99.9 ACRES',
    'RESIDENTIAL COMMON AREA W/IMPS', 
    'RESIDENTIAL COMMON AREA/STREETS', 'RESIDENTIAL EXCEPTIONAL', 
    'RESIDENTIAL RESTRICTED', 'VACANT LAND - PREDOMINATE RESIDENTIAL USE'
]

# Filter for residential property types and clean data
filtered_data = data.filter(
    (col("property_type").isin(residential_types))
)

# Show a sample of the filtered data
filtered_data.show(10)

# Save the filtered and selected data to a new CSV
output_path = "data_residential"
filtered_data.write.csv(output_path, header=True, mode="overwrite", compression = "gzip")
