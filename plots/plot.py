import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import matplotlib.ticker as mtick

def main(in_directory, out_directory):
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Zoning Land Value and Tax Analysis").getOrCreate()

    # Load data
    df = spark.read.csv(in_directory, header=True, inferSchema=True)

    # Create output directory if it doesn't exist
    os.makedirs(out_directory, exist_ok=True)

    # Plot 1: Impact of Land Value on Tax Levy (remove outliers)
    filtered_df = df.filter(col("CURRENT_LAND_VALUE") < 1e6)  # Removed outliers
    land_tax_pd = (
        filtered_df.groupBy("CURRENT_LAND_VALUE")
        .agg(avg("TAX_LEVY").alias("avg_tax_levy"))
        .toPandas()
    )

    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=land_tax_pd, x="CURRENT_LAND_VALUE", y="avg_tax_levy", color="blue", alpha=0.6)
    plt.title("Impact of Land Value on Tax Levy", fontsize=16)
    plt.xlabel("Current Land Value ($)", fontsize=14)
    plt.ylabel("Average Tax Levy ($)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_directory}/land_vs_tax_no_outliers.png")
    plt.close()

    # Plot 2: Average Land Value by Zoning District (removed x-axis labels)
    zoning_land_pd = (
        df.groupBy("ZONING_DISTRICT")
        .agg(avg("CURRENT_LAND_VALUE").alias("avg_land_value"))
        .orderBy("avg_land_value", ascending=False)
        .toPandas()
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(data=zoning_land_pd, x="ZONING_DISTRICT", y="avg_land_value", palette="viridis")
    plt.title("Average Land Value by Zoning District", fontsize=16)
    plt.xlabel("Zoning District (Hidden for Clarity)", fontsize=14)
    plt.ylabel("Average Land Value ($)", fontsize=14)
    plt.xticks([], [])  # Hide x-axis labels
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x / 1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(f"{out_directory}/zoning_vs_land_value_no_xlabels.png")
    plt.close()

    # Plot 3: Average Tax Levy by Zoning District (Top 50, removed x-axis labels)
    zoning_tax_pd = (
        df.groupBy("ZONING_DISTRICT")
        .agg(avg("TAX_LEVY").alias("avg_tax_levy"))
        .orderBy("avg_tax_levy", ascending=False)
        .limit(50)  # Show top 50 zoning districts
        .toPandas()
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(data=zoning_tax_pd, x="ZONING_DISTRICT", y="avg_tax_levy", palette="Blues")
    plt.title("Average Tax Levy by Zoning District (Top 50)", fontsize=16)
    plt.xlabel("Zoning District (Hidden for Clarity)", fontsize=14)
    plt.ylabel("Average Tax Levy ($)", fontsize=14)
    plt.xticks([], [])  # Hide x-axis labels
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x / 1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(f"{out_directory}/zoning_vs_tax_top50.png")
    plt.close()

    print(f"Plots saved at {out_directory}")

    spark.stop()

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
