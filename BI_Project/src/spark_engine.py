from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, sum as _sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import os
import json

# ---------------------------------------------------------
# Step 1: Initialize PySpark with Kafka Integration
# ---------------------------------------------------------
print("⏳ Initializing PySpark Engine with Kafka connectors...")

spark = SparkSession.builder \
    .appName("RealTimeBusinessIntelligence") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Reduce terminal noise by setting logging to WARN
spark.sparkContext.setLogLevel("WARN")

# Define the exact schema matching your Kafka Producer
sales_schema = StructType([
    StructField("OrderID", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Region", StringType(), True),
    StructField("Sales", DoubleType(), True)
])

# Read the live data stream from Kafka
kafka_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "superstore_sales") \
    .option("startingOffsets", "latest") \
    .load()

# Parse the JSON string coming from Kafka
parsed_stream = kafka_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), sales_schema).alias("data")) \
    .select("data.*")

# Group by Region and sum the sales
live_aggregation = parsed_stream.groupBy("Region").agg(_sum("Sales").alias("Total_Revenue"))

# ---------------------------------------------------------
# Step 2: Bulletproof JSON Writer for Windows
# ---------------------------------------------------------
def write_to_json(batch_df, batch_id):
    # Collect the data natively without Pandas to avoid PySpark memory crashes
    rows = batch_df.collect()
    
    if rows:
        # Convert Spark rows to a standard Python list of dictionaries
        data_list = [row.asDict() for row in rows]
        
        # FORCE the file to save in the exact same folder as this script (src folder)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, 'live_sales.json')
        
        # Write to JSON using standard Python
        with open(save_path, 'w') as f:
            json.dump(data_list, f)
            
        print(f"✅ Batch {batch_id} successfully written to: {save_path}")

print("📡 Waiting for live sales data... Saving to JSON!")

# Start the streaming query
query = live_aggregation.writeStream \
    .outputMode("complete") \
    .foreachBatch(write_to_json) \
    .start()

query.awaitTermination()