from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType
from pyspark.sql.functions import collect_list, udf, col
from scipy.stats import f_oneway
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#use data Partioned!!

# Defining the schema
schema = StructType([
    StructField("PID", StringType(), True),                             # Property Identifier
    StructField("LEGAL_TYPE", StringType(), True),                     # Legal type of the property
    StructField("FOLIO", StringType(), True),                          # Folio number
    StructField("LAND_COORDINATE", StringType(), True),                # Land coordinate
    StructField("ZONING_DISTRICT", StringType(), True),                # Zoning district
    StructField("ZONING_CLASSIFICATION", StringType(), True),          # Zoning classification
    StructField("LOT", StringType(), True),                            # Lot
    StructField("PLAN", StringType(), True),                           # Plan
    StructField("BLOCK", StringType(), True),                          # Block
    StructField("DISTRICT_LOT", StringType(), True),                   # District lot
    StructField("FROM_CIVIC_NUMBER", StringType(), True),              # From civic number
    StructField("TO_CIVIC_NUMBER", StringType(), True),                # To civic number
    StructField("STREET_NAME", StringType(), True),                    # Street name
    StructField("PROPERTY_POSTAL_CODE", StringType(), True),           # Property postal code
    StructField("NARRATIVE_LEGAL_LINE1", StringType(), True),          # Narrative legal description line 1
    StructField("NARRATIVE_LEGAL_LINE2", StringType(), True),          # Narrative legal description line 2
    StructField("NARRATIVE_LEGAL_LINE3", StringType(), True),          # Narrative legal description line 3
    StructField("NARRATIVE_LEGAL_LINE4", StringType(), True),          # Narrative legal description line 4
    StructField("NARRATIVE_LEGAL_LINE5", StringType(), True),          # Narrative legal description line 5
    StructField("CURRENT_LAND_VALUE", FloatType(), True),              # Current land value
    StructField("CURRENT_IMPROVEMENT_VALUE", FloatType(), True),       # Current improvement value
    StructField("TAX_ASSESSMENT_YEAR", StringType(), True),            # Tax assessment year
    StructField("PREVIOUS_LAND_VALUE", FloatType(), True),             # Previous land value
    StructField("PREVIOUS_IMPROVEMENT_VALUE", FloatType(), True),      # Previous improvement value
    StructField("YEAR_BUILT", StringType(), True),                     # Year built
    StructField("BIG_IMPROVEMENT_YEAR", StringType(), True),           # Big improvement year
    StructField("TAX_LEVY", DoubleType(), True),                       # Tax levy
    StructField("NEIGHBOURHOOD_CODE", StringType(), True),             # Neighborhood code
    StructField("REPORT_YEAR", StringType(), True)                     # Report year
])

# Hardcoded neighbourhood codes which I got from the selecting the distinct neighbourhood code
#unique_neighbourhood_codes = data.select("NEIGHBOURHOOD_CODE").distinct()
#unique_neighbourhood_codes.show(truncate=False)
neighbourhood_codes = [
    "030", "009", "028", "012", "027", "013", "024", "015",
    "006", "019", "020", "011", "025", "003", "005", "016",
    "029", "018", "008", "022", "001", "014", "010", "023",
    "004", "017", "007", "026", "021", "002"
]


def main(in_directory):
    
    spark = SparkSession.builder.appName("ANOVA on Hardcoded Neighbourhoods").getOrCreate()

    # reading the data with a semicolon delimter
    data = spark.read.csv(in_directory, schema=schema, header=True, sep=";")
    #filter the null values

    filtered_data = data.filter(col("PREVIOUS_LAND_VALUE").isNotNull())

    # group by NEIGHBOURHOOD_CODE and collect land values into a list
    grouped_data = filtered_data.groupBy("NEIGHBOURHOOD_CODE").agg(
        collect_list("PREVIOUS_LAND_VALUE").alias("land_values")
    )

    # collect data for ANOVA
    grouped_values = grouped_data.select("land_values").rdd.map(lambda row: row["land_values"]).collect()

    #  ANOVA
    try:
        _, p_value = f_oneway(*grouped_values)
        print(f"ANOVA Test Result: p-value = {p_value}")
        if p_value < 0.05:
            print("The means of PREVIOUS_LAND_VALUE are significantly different among NEIGHBOURHOOD_CODE groups.")
        else:
            print("No significant difference in means of PREVIOUS_LAND_VALUE among NEIGHBOURHOOD_CODE groups.")
    except Exception as e:
        print(f"ANOVA computation failed: {e}")
    #since the p val was less than 0.05 we can do tukey test
    # convert grouped data  for Tukey HSD
    flat_data = filtered_data.select("NEIGHBOURHOOD_CODE", "PREVIOUS_LAND_VALUE").collect()

    # preparing the data for tukey 
    neighbourhood_codes = []
    land_values = []
    for row in flat_data:
        neighbourhood_codes.append(row["NEIGHBOURHOOD_CODE"])
        land_values.append(row["PREVIOUS_LAND_VALUE"])

    # perform tukey 
    try:
        tukey_result = pairwise_tukeyhsd(
            endog=land_values,  
            groups=neighbourhood_codes,  
            alpha=0.05
        )
        print("\nTukey HSD Test Results:")
        print(tukey_result)
    except Exception as e:
        print(f"Tukey HSD computation failed: {e}")

if __name__ == '__main__':
    in_directory = sys.argv[1]
    main(in_directory)
