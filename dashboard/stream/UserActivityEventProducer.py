import random
import time
from confluent_kafka import Producer
from confluent_kafka.serialization import SerializationContext, MessageField
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

# Utility to generate user activity
def generate_user_activity():
    activities = ["Click", "Purchase"]
    return {
        "id": random.randint(1, 1000),
        "campaignid": random.randint(1, 20),
        "orderid": random.randint(1, 10000),
        "total_amount": random.randint(1, 10000),
        "units": random.randint(1, 100),
        "tags": {"activity": random.choice(activities)},
    }

# Serialization callback for AvroSerializer
def user_activity_to_dict(activity, ctx):
    return activity

def main():
    # Kafka Producer configuration
    producer_config = {
        "bootstrap.servers": "localhost:9092",
    }

    # Schema Registry configuration
    schema_registry_config = {
        "url": "http://localhost:8081",
    }

    # Initialize Schema Registry client
    schema_registry_client = SchemaRegistryClient(schema_registry_config)

    # Load Avro schema
    schema_path = "consumer_activity.avsc"
    with open(schema_path, "r") as schema_file:
        avro_schema_str = schema_file.read()

    # Initialize AvroSerializer
    value_serializer = AvroSerializer(
        schema_registry_client,
        avro_schema_str,
        user_activity_to_dict
    )

    # Initialize Kafka Producer
    producer = Producer(producer_config)

    topic = "consumer_activity"

    while True:
        # Generate user activity
        activity = generate_user_activity()

        # Generate random key and encode in UTF-8
        key = str(random.randint(1, 10000))

        try:
            # Serialize value
            serialized_value = value_serializer(activity, SerializationContext(topic, MessageField.VALUE))

            # Produce message to Kafka
            producer.produce(topic=topic, key=key, value=serialized_value)
            producer.flush()
            print(f"Produced record: Key={key}, Value={activity}")
        except Exception as e:
            print(f"Failed to produce record: {e}")

        time.sleep(1)

if __name__ == "__main__":
    main()
