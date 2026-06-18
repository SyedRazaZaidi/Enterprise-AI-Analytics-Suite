import pandas as pd
from kafka import KafkaProducer
import json
import time
import os

try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print("🚦 Connected to Kafka Broker successfully!")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit()

topic_name = 'superstore_sales'
print(f"🚀 Starting Live Sales Stream to topic: '{topic_name}'...")

# Load the CSV Data
csv_path = os.path.join(os.path.dirname(__file__), '../data/sales.csv')
df = pd.read_csv(csv_path)

try:
    # Stream data infinitely to simulate a 24/7 business
    while True:
        for index, row in df.iterrows():
            data_dict = row.to_dict()
            producer.send(topic_name, value=data_dict)
            print(f"Sent: {data_dict}")
            time.sleep(2)
except KeyboardInterrupt:
    print("\n🛑 Sales Streaming Paused by User.")
    producer.close()