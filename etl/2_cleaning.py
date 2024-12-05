import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType, FloatType, DoubleType)
from pyspark.sql.functions import col, trim, when, mean as _mean

def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("Spark_Only_Cleaning").getOrCreate()
    spark.conf.set("spark.sql.debug.maxToStringFields", 2000)

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

    # Read data
    df = spark.read.csv(in_directory, schema=schema, header=True, sep=";")

    # Clean data
    # Trim spaces and convert empty to null
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))
        df = df.withColumn(c, when(col(c) == "", None).otherwise(col(c)))

    # Drop duplicates and rows without PID
    df = df.dropDuplicates().filter(col("PID").isNotNull())

    # Convert YEAR_BUILT and BIG_IMPROVEMENT_YEAR to int
    df = df.withColumn("YEAR_BUILT_INT", col("YEAR_BUILT").cast("int")) \
           .withColumn("BIG_IMPROVEMENT_YEAR_INT", col("BIG_IMPROVEMENT_YEAR").cast("int")) \
           .withColumn("REPORT_YEAR_INT", col("REPORT_YEAR").cast("int"))

    # Impute numeric columns
    numeric_cols = ["CURRENT_LAND_VALUE", "CURRENT_IMPROVEMENT_VALUE", 
                    "PREVIOUS_LAND_VALUE", "PREVIOUS_IMPROVEMENT_VALUE", "TAX_LEVY"]
    from pyspark.ml.feature import Imputer
    imputer = Imputer(strategy="mean", inputCols=numeric_cols, outputCols=[c + "_imp" for c in numeric_cols])
    df = imputer.fit(df).transform(df)
    for c in numeric_cols:
        df = df.drop(c).withColumnRenamed(c + "_imp", c)

    # Fill categorical columns with 'UNKNOWN'
    cat_cols = ["LEGAL_TYPE", "ZONING_DISTRICT", "ZONING_CLASSIFICATION", "LOT", "PLAN", "BLOCK",
                "DISTRICT_LOT", "FROM_CIVIC_NUMBER", "TO_CIVIC_NUMBER", "STREET_NAME", 
                "PROPERTY_POSTAL_CODE", "NARRATIVE_LEGAL_LINE1", "NARRATIVE_LEGAL_LINE2",
                "NARRATIVE_LEGAL_LINE3", "NARRATIVE_LEGAL_LINE4", "NARRATIVE_LEGAL_LINE5",
                "NEIGHBOURHOOD_CODE", "TAX_ASSESSMENT_YEAR", "REPORT_YEAR", "YEAR_BUILT", "BIG_IMPROVEMENT_YEAR"]
    for c in cat_cols:
        df = df.na.fill("UNKNOWN", subset=[c])

    # Feature engineering: property_age, improvement_gap
    df = df.withColumn("property_age", 
                       when(col("YEAR_BUILT_INT").isNotNull() & col("REPORT_YEAR_INT").isNotNull(),
                            col("REPORT_YEAR_INT") - col("YEAR_BUILT_INT")).otherwise(None))
    df = df.withColumn("improvement_gap",
                       when(col("BIG_IMPROVEMENT_YEAR_INT").isNotNull() & col("YEAR_BUILT_INT").isNotNull(),
                            col("BIG_IMPROVEMENT_YEAR_INT") - col("YEAR_BUILT_INT")).otherwise(None))

    # Fill nulls in property_age and improvement_gap with 0 to avoid null issues later
    df = df.na.fill(0, subset=["property_age","improvement_gap"])

    # Write the cleaned output
    df.write.csv(out_directory, mode="overwrite", header=True, compression="gzip")
    print("Data cleaned and written to:", out_directory)

    spark.stop()

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
