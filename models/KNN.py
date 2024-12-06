import sys
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main(in_directory, out_directory):
    spark = SparkSession.builder.appName("Enhanced_KNN_Classifier").getOrCreate()

    
    df = spark.read.csv(in_directory, header=True, inferSchema=True).dropDuplicates()

    # strip column names
    df = df.select([col(c).alias(c.strip()) for c in df.columns])

    # splitting the numerical and categorical columns and doing the necessary transformations on both
    numerical_columns = [
        "CURRENT_LAND_VALUE",
        "CURRENT_IMPROVEMENT_VALUE",
        "PREVIOUS_IMPROVEMENT_VALUE",
        "property_age"
    ]  

    # convert numerical columns to double
    for col_name in numerical_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    # encode target column
    label_indexer = StringIndexer(inputCol="CURRENT_LAND_VALUE", outputCol="label")
    df = label_indexer.fit(df).transform(df)

    # encode categorical columns using stringindexer
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

    # combine features
    feature_cols = (
        [col_name for col_name in numerical_columns if col_name in df.columns]
        + [f"{col_name}_index" for col_name in categorical_cols if f"{col_name}_index" in df.columns]
    )
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    df = assembler.transform(df)

    # convert to Pandas
    df_pandas = df.select(*numerical_columns, "label").toPandas()

    # extract features and target
    X = pd.DataFrame(df.select("features_raw").toPandas()["features_raw"].tolist())  
    y = df_pandas["label"]  # Use the encoded 'label' column as the target

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # predictions
    y_pred = knn.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Classifier Accuracy: {accuracy:.4f}")

    #  predictions
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv(f"{out_directory}/predictions.csv", index=False)

    # plots for predictions
    predictors_to_plot = ["CURRENT_IMPROVEMENT_VALUE", "property_age"]
    for predictor in predictors_to_plot:
        if predictor in df_pandas.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(df_pandas[predictor], df_pandas["label"], alpha=0.5)
            plt.xlabel(predictor)
            plt.ylabel("Encoded CURRENT_LAND_VALUE (label)")
            plt.title(f"Distribution of Encoded CURRENT_LAND_VALUE by {predictor}")
            plt.grid()
            plt.savefig(f"{out_directory}/{predictor}_distribution.png")

    spark.stop()

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
