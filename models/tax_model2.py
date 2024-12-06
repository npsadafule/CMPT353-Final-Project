import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, PolynomialExpansion
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("Polynomial_Tax_Levy_Prediction").getOrCreate()

    # Load data
    df = spark.read.csv(in_directory, header=True, inferSchema=True).dropDuplicates()

    # Cast numerical columns to correct types
    numerical_columns = [
        "CURRENT_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_LAND_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "property_age",
    ]

    for col_name in numerical_columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # Log transformation for skewed features
    for col_name in numerical_columns[:4]:  # Only financial columns
        df = df.withColumn(f"log_{col_name}", log1p(col(col_name)))

    # Filter rows where TAX_LEVY > 0 and drop null values
    train_df = df.filter(col("TAX_LEVY") > 0).na.drop()

    # Encode categorical columns
    categorical_cols = ["ZONING_DISTRICT", "NEIGHBOURHOOD_CODE"]
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        train_df = indexer.fit(train_df).transform(train_df)

    # Define feature columns
    feature_cols = (
        [f"log_{col_name}" for col_name in numerical_columns[:4]]  # Log-transformed columns
        + numerical_columns[4:]  # property_age
        + [f"{col_name}_index" for col_name in categorical_cols]  # Encoded categorical features
    )

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    train_df = assembler.transform(train_df)

    # Polynomial features
    poly_expansion = PolynomialExpansion(inputCol="features_raw", outputCol="features_poly", degree=2)
    train_df = poly_expansion.transform(train_df)

    # Scale features
    scaler = StandardScaler(inputCol="features_poly", outputCol="features", withStd=True, withMean=True)
    train_df = scaler.fit(train_df).transform(train_df)

    # Train-test split
    train_data, test_data = train_df.randomSplit([0.8, 0.2], seed=13)

    # Train Linear Regression model with L2 regularization
    lr = LinearRegression(
        featuresCol="features",
        labelCol="TAX_LEVY",
        predictionCol="prediction",
        regParam=0.1,  # L2 regularization strength
        elasticNetParam=0.0,  # Pure L2 regularization
    )
    lr_model = lr.fit(train_data)

    # Evaluate model
    predictions = lr_model.transform(test_data)
    predictions.select("TAX_LEVY", "prediction", "features").show()

    # Save predictions
    predictions.select("TAX_LEVY", "prediction").write.csv(out_directory, mode="overwrite", header=True)

    # Evaluate metrics
    evaluator = RegressionEvaluator(labelCol="TAX_LEVY", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    # Print evaluation metrics
    print(f"Root Mean Square Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² (Coefficient of Determination): {r2}")
    print("Model coefficients:", lr_model.coefficients)
    print("Model intercept:", lr_model.intercept)

    # Visualization
    actual_values = predictions.select("TAX_LEVY").rdd.flatMap(lambda x: x).collect()
    predicted_values = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
    residuals = [a - p for a, p in zip(actual_values, predicted_values)]

    # Actual vs Predicted Tax Levy plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predicted_values, alpha=0.6, label="Predicted Points")
    plt.plot([0, max(actual_values)], [0, max(actual_values)], color="red", linestyle="--", label="Ideal Fit")
    plt.title("Actual vs Predicted Tax Levy (Polynomial)")
    plt.xlabel("Actual Tax Levy")
    plt.ylabel("Predicted Tax Levy")
    plt.ylim(0, 10000)  # Limit y-axis to 10,000
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_directory}/actual_vs_predicted_polynomial.png")
    plt.show()

    # Residuals histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color="blue", alpha=0.6)
    plt.title("Residuals Distribution (Polynomial)")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_directory}/residuals_histogram_polynomial.png")
    plt.show()

    spark.stop()


if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
