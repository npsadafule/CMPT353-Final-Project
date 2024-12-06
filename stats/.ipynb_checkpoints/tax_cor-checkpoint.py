from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

def main(in_directory, out_directory):
    # received this code from exercises 
    spark = SparkSession.builder.appName("Feature_Importance").getOrCreate()

    # here we are just removing output directory if it already exists to allow overwriting
    if os.path.exists(out_directory):
        shutil.rmtree(out_directory)
    os.makedirs(out_directory)

    df = spark.read.csv(in_directory, header=True, inferSchema=True)
    df = df.dropDuplicates().na.drop() # some filtering
    # only choosing what we need
    numerical_columns = [
        "CURRENT_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_LAND_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "property_age",
        "improvement_gap"
    ]
    categorical_columns = ["LEGAL_TYPE", "ZONING_DISTRICT", "NEIGHBOURHOOD_CODE"]

    # distinct values in categorical columns
    print("Distinct values in categorical columns:")
    for col_name in categorical_columns:
        distinct_count = df.select(col_name).distinct().count()
        print(f"{col_name}: {distinct_count} distinct values")

    # encode categorical columns
    for col_name in categorical_columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        df = indexer.fit(df).transform(df)

    # combine all features
    feature_cols = numerical_columns + [f"{col_name}_index" for col_name in categorical_columns]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Random Forest model with adjusted maxBins
    rf = RandomForestRegressor(featuresCol="features", labelCol="TAX_LEVY", maxBins=50)  # increased maxBins
    rf_model = rf.fit(df)

    # Extract feature importances
    importances = rf_model.featureImportances.toArray()
    feature_names = numerical_columns + categorical_columns
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # writing to csv
    feature_importance.to_csv(f"{out_directory}/feature_importance.csv", index=False)

    # plotting results
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{out_directory}/feature_importance.png")
    plt.close()

    print("Feature importance analysis complete. Results saved to:", out_directory)
    spark.stop()

if __name__ == "__main__":
    import sys
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
