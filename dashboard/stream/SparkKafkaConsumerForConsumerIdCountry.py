from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.streaming import Trigger
import json


def main():
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .master("local") \
            .appName("realtime spark kafka consumer") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        # Define Kafka source
        input_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "http://localhost:9092") \
            .option("subscribe", "consumer_activity") \
            .option("group.id", "user_activity_grp") \
            .option("schema.registry.url", "http://localhost:8081") \
            .load()

        # Load Avro schema
        with open("./src/main/resources/userActivity.avsc", "r") as schema_file:
            json_format_schema = schema_file.read()

        # Deserialize Kafka messages using Avro schema
        from_avro_config = {
            "schema": json.loads(json_format_schema),
            "schema.registry.url": "http://localhost:8081"
        }

        df2 = input_stream.selectExpr(
            "from_avro(value, '{schema}') as activity".format(schema=json_format_schema)
        )

        # Load demographic data from MySQL
        demographic_data = spark.read.format("jdbc") \
            .option("url", "jdbc:mysql://localhost:3307") \
            .option("dbtable", "users.userdemographics") \
            .option("user", "root") \
            .option("password", "example") \
            .load()

        demographic_data.show()

        # Join Kafka stream with MySQL data
        df3 = df2.join(
            demographic_data,
            df2.col("activity.id") == demographic_data.col("i")
        )

        # Group by country and count
        df4 = df3.groupBy("country").count()

        # Write to console
        query = df4.writeStream \
            .outputMode("complete") \
            .format("console") \
            .start()

        # Write to MySQL using foreachBatch
        def write_to_mysql(batch_df, batch_id):
            print("inside foreachBatch")
            batch_df.write \
                .format("jdbc") \
                .option("url", "jdbc:mysql://localhost:3307") \
                .option("dbtable", "users.countryAgg") \
                .option("user", "root") \
                .option("password", "example") \
                .mode("overwrite") \
                .save()

        query2 = df4.writeStream \
            .outputMode("complete") \
            .trigger(Trigger.ProcessingTime("10 seconds")) \
            .foreachBatch(write_to_mysql) \
            .start()

        query2.awaitTermination()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
