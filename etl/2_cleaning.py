import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType, FloatType, DoubleType)
from pyspark.sql.functions import col, trim, when, mean as _mean
from pyspark.ml.feature import Imputer

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

    # Trim spaces and convert empty strings to null
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))
        df = df.withColumn(c, when(col(c) == "", None).otherwise(col(c)))

    # Drop duplicates and rows without PID
    df = df.dropDuplicates().filter(col("PID").isNotNull())

    # Drop rows where required fields are null
    required_fields = [
        "CURRENT_LAND_VALUE", 
        "PREVIOUS_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "TAX_LEVY",
        "YEAR_BUILT", 
        "BIG_IMPROVEMENT_YEAR",
        "ZONING_DISTRICT", 
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
        "TAX_ASSESSMENT_YEAR", 
        "REPORT_YEAR"
    ]
    for field in required_fields:
        df = df.filter(col(field).isNotNull())

    # Convert YEAR_BUILT, BIG_IMPROVEMENT_YEAR, and REPORT_YEAR to integers
    df = df.withColumn("YEAR_BUILT", col("YEAR_BUILT").cast("int")) \
           .withColumn("BIG_IMPROVEMENT_YEAR", col("BIG_IMPROVEMENT_YEAR").cast("int")) \
           .withColumn("REPORT_YEAR", col("REPORT_YEAR").cast("int"))

    # Handle outliers in numeric fields using IQR
    numeric_cols = ["CURRENT_LAND_VALUE", "PREVIOUS_LAND_VALUE"]
    for field in numeric_cols:
        q1, q3 = df.approxQuantile(field, [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Cap outliers to upper and lower bounds
        df = df.withColumn(
            field,
            when(col(field) > upper_bound, upper_bound)
            .when(col(field) < lower_bound, lower_bound)
            .otherwise(col(field))
        )

    # Impute missing numeric columns with mean
    impute_cols = ["CURRENT_IMPROVEMENT_VALUE", "PREVIOUS_IMPROVEMENT_VALUE", "TAX_LEVY"]
    imputer = Imputer(strategy="mean", inputCols=impute_cols, outputCols=[c + "_imp" for c in impute_cols])
    df = imputer.fit(df).transform(df)
    for c in impute_cols:
        df = df.drop(c).withColumnRenamed(c + "_imp", c)

    # Feature engineering: property_age, improvement_gap
    df = df.withColumn("property_age", 
                       when(col("YEAR_BUILT").isNotNull() & col("REPORT_YEAR").isNotNull(),
                            col("REPORT_YEAR") - col("YEAR_BUILT")).otherwise(None))
    df = df.withColumn("improvement_gap",
                       when(col("BIG_IMPROVEMENT_YEAR").isNotNull() & col("YEAR_BUILT").isNotNull(),
                            col("BIG_IMPROVEMENT_YEAR") - col("YEAR_BUILT")).otherwise(None))

    # Fill nulls in property_age and improvement_gap with 0 to avoid null issues later
    df = df.na.fill(0, subset=["property_age", "improvement_gap"])

    # Write the cleaned output
    df.write.csv(out_directory, mode="overwrite", header=True, compression="gzip")
    print("Data cleaned and written to:", out_directory)

    spark.stop()

if __name__ == '__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
