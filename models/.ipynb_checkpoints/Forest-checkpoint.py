import sys
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("Enhanced_Tax_Levy_Regression").getOrCreate()

   
    df = spark.read.csv(in_directory, header=True, inferSchema=True).dropDuplicates()

    # striping columns
    df = df.select([col(c).alias(c.strip()) for c in df.columns])

    # splitting the numerical and categorical columns and doing the necessary transformations on both
    numerical_columns = [
        "CURRENT_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "property_age"
    ]  
    
    #had slight issue with columns present so this was added
    for col_name in numerical_columns:
        if col_name in df.columns:  
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
        else:
            print(f"Warning: Column '{col_name}' not found in the dataset!")

    # log-transform skewed numerical features
    for col_name in numerical_columns[1:]:  
        if col_name in df.columns:
            df = df.withColumn(f"log_{col_name}", log1p(col(col_name)))

    # remove missing values in key columns
    key_columns = ["CURRENT_LAND_VALUE"] + numerical_columns[1:]
    key_columns = [col for col in key_columns if col in df.columns]  
    df = df.na.drop(subset=key_columns)

    # encoding categorical columns using stringindexer
    categorical_cols = [
        "ZONING_DISTRICT",
        "NEIGHBOURHOOD_CODE",
        "YEAR_BUILT",
        "BIG_IMPROVEMENT_YEAR",
        "ZONING_CLASSIFICATION"
    ]
    for col_name in categorical_cols:
        if col_name in df.columns:
            indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
            df = indexer.fit(df).transform(df)
        else:
            print(f"Warning: Column '{col_name}' not found in the dataset!")

    # combining features for regression
    feature_cols = (
        [f"log_{col_name}" for col_name in numerical_columns[1:] if f"log_{col_name}" in df.columns]
        + [f"{col_name}_index" for col_name in categorical_cols if f"{col_name}_index" in df.columns]
    )
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    df = assembler.transform(df)

    # scaleing features as taught in class
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    df = scaler.fit(df).transform(df)

    # train-test split into 80 and 20
    train_data, test_data = df.randomSplit([0.8, 0.2])

    # training random forest regressor
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="CURRENT_LAND_VALUE",
        predictionCol="prediction",
        numTrees=10,
        maxDepth=5
    )
    rf_model = rf.fit(train_data)

    # predictions
    predictions = rf_model.transform(test_data)
    predictions.select("CURRENT_LAND_VALUE", "prediction", "features").show()

    #evaluations
    evaluator = RegressionEvaluator(labelCol="CURRENT_LAND_VALUE", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    print(f"Random Forest Regressor RMSE: {rmse:.4f}")
    print(f"Random Forest Regressor MAE: {mae:.4f}")
    print(f"Random Forest Regressor R²: {r2:.4f}")
    feature_importances = rf_model.featureImportances.toArray()

    #predictions
    predictions.select("CURRENT_LAND_VALUE", "prediction").write.csv(out_directory, mode="overwrite", header=True)

    # plots for report
    predictions_pd = predictions.select("CURRENT_LAND_VALUE", "prediction").toPandas()

    #  actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_pd["CURRENT_LAND_VALUE"], predictions_pd["prediction"], alpha=0.5)
    plt.xlabel("Actual Land Value")
    plt.ylabel("Predicted Land Value")
    plt.title("Actual vs Predicted Land Value")
    plt.grid()
    plt.savefig("actual_vs_predicted.png")

    #  residuals
    residuals = predictions_pd["CURRENT_LAND_VALUE"] - predictions_pd["prediction"]
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid()
    plt.savefig("residuals_distribution.png")

    #  feature importances
    plt.figure(figsize=(12, 8))
    feature_names = feature_cols  # List of feature names
    sorted_idx = feature_importances.argsort()
    plt.barh([feature_names[i] for i in sorted_idx], feature_importances[sorted_idx], color='skyblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance from Random Forest")
    plt.grid(axis='x')
    plt.savefig("feature_importance.png")

    spark.stop()

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
