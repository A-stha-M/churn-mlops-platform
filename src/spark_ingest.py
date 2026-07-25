"""
Phase 1a — Spark Data Ingestion & Cleaning for the Telco Churn dataset.
Run: python3 src/spark_ingest.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ChurnDataIngestion")
        .master("local[*]")
        .getOrCreate()
    )


def load_and_clean(spark: SparkSession, csv_path: str):
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    print("=== Spark-inferred schema ===")
    df.printSchema()
    print(f"\nRow count: {df.count()}")

    # TotalCharges arrives as a string because 11 rows (tenure=0, brand new
    # customers never billed yet) have a blank value instead of a number.
    df = df.withColumn(
        "TotalCharges_clean",
        F.when(F.trim(F.col("TotalCharges")) == "", None)
         .otherwise(F.col("TotalCharges"))
         .cast(DoubleType())
    )
    df = df.withColumn(
        "TotalCharges_clean",
        F.when(F.col("TotalCharges_clean").isNull(), 0.0)
         .otherwise(F.col("TotalCharges_clean"))
    )

    df = (
        df
        .withColumnRenamed("customerID", "customer_id")
        .withColumnRenamed("SeniorCitizen", "senior_citizen")
        .withColumnRenamed("Partner", "partner")
        .withColumnRenamed("Dependents", "dependents")
        .withColumnRenamed("PhoneService", "phone_service")
        .withColumnRenamed("MultipleLines", "multiple_lines")
        .withColumnRenamed("InternetService", "internet_service")
        .withColumnRenamed("OnlineSecurity", "online_security")
        .withColumnRenamed("OnlineBackup", "online_backup")
        .withColumnRenamed("DeviceProtection", "device_protection")
        .withColumnRenamed("TechSupport", "tech_support")
        .withColumnRenamed("StreamingTV", "streaming_tv")
        .withColumnRenamed("StreamingMovies", "streaming_movies")
        .withColumnRenamed("Contract", "contract_type")
        .withColumnRenamed("PaperlessBilling", "paperless_billing")
        .withColumnRenamed("PaymentMethod", "payment_method")
        .withColumnRenamed("MonthlyCharges", "monthly_charges")
        .drop("TotalCharges")
        .withColumnRenamed("TotalCharges_clean", "total_charges")
        # Spark's SQL analyzer is case-insensitive by default, so we must
        # rename "Churn" before deriving a "churn" column, or .drop("Churn")
        # would silently drop both.
        .withColumnRenamed("Churn", "churn_raw")
        .withColumn("churn", F.when(F.col("churn_raw") == "Yes", 1).otherwise(0))
        .drop("churn_raw")
    )
    return df


def explore(df):
    print("\n=== Churn rate overall ===")
    df.groupBy("churn").count().show()

    print("=== Churn rate by contract type ===")
    (
        df.groupBy("contract_type")
        .agg(F.count("*").alias("customers"), F.round(F.avg("churn"), 3).alias("churn_rate"))
        .orderBy("churn_rate", ascending=False)
        .show()
    )


if __name__ == "__main__":
    spark = get_spark_session()
    df = load_and_clean(spark, "data/raw_telco_churn.csv")
    explore(df)

    pandas_df = df.toPandas()
    pandas_df.to_csv("data/spark_cleaned_churn.csv", index=False)
    print(f"\nConverted to pandas: {pandas_df.shape}")
    print("Saved -> data/spark_cleaned_churn.csv")

    spark.stop()