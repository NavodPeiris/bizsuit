import random
import time
from confluent_kafka import Producer
from fastavro.schema import load_schema
from fastavro import writer
from io import BytesIO
import json

# Utility to generate user activity (simulating UserActivityUtil)
def generate_user_activity():
    activities = ["Click", "Purchase"]
    return {
        "id": random.randint(1, 10000),
        "campaignid": random.randint(1, 20),
        "orderid": random.randint(1, 10000),
        "total_amount": random.randint(1, 10000),
        "units": random.randint(1, 100),
        "tags": {"activity": random.choice(activities)},
    }

def main():
    # Kafka producer configuration
    producer_config = {
        "bootstrap.servers": "http://localhost:9092",
        "schema.registry.url": "http://localhost:8081",
        "key.serializer": "org.apache.kafka.common.serialization.StringSerializer",
        "value.serializer": "io.confluent.kafka.serializers.KafkaAvroSerializer",
    }
    producer = Producer(producer_config)

    topic = "consumer_activity"

    # Load Avro schema
    schema_path = "./consumer_activity.avsc"  # Ensure schema is saved here
    schema = load_schema(schema_path)

    while True:
        # Generate user activity
        activity = generate_user_activity()

        # Serialize to Avro
        avro_bytes = BytesIO()
        writer(avro_bytes, schema, [activity])
        avro_bytes.seek(0)

        # Generate random key
        key = str(random.randint(1, 10000))

        # Produce message to Kafka
        try:
            producer.produce(topic, key=key, value=avro_bytes.getvalue())
            producer.flush()
            print(f"Produced record: Key={key}, Value={json.dumps(activity)}")
        except Exception as e:
            print(f"Failed to produce record: {e}")

        time.sleep(1)

if __name__ == "__main__":
    main()
