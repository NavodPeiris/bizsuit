from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, struct, expr
from pyspark.sql.streaming import StreamingQueryException
from pyspark.sql.types import MapType, StringType, IntegerType, StructType

def main():
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .master("local") \
            .appName("realtime spark kafka consumer") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        # Define Kafka source
        user_activity_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "http://localhost:9092") \
            .option("subscribe", "consumer_activity") \
            .option("group.id", "user_activity_grp") \
            .option("schema.registry.url", "http://localhost:8081") \
            .load()

        # Define schemas
        user_activity_schema = StructType() \
            .add("id", IntegerType()) \
            .add("campaignid", IntegerType()) \
            .add("orderid", IntegerType()) \
            .add("total_amount", IntegerType()) \
            .add("units", IntegerType()) \
            .add("tags", MapType(StringType(), StringType()))

        user_activity_demographic_schema = StructType() \
            .add("id", IntegerType()) \
            .add("orderid", IntegerType()) \
            .add("campaignid", IntegerType()) \
            .add("total_amount", IntegerType()) \
            .add("units", IntegerType()) \
            .add("age", IntegerType()) \
            .add("tags", MapType(StringType(), StringType()))

        # Deserialize Kafka messages using schema
        user_activity_stream_des = user_activity_stream.selectExpr(
            "CAST(value AS STRING) as json_string"
        ).select(
            expr(f"from_json(json_string, '{user_activity_schema.json()}')").alias("activity")
        )

        # Load demographic data from MySQL
        demographic_data = spark.read.format("jdbc") \
            .option("url", "jdbc:mysql://localhost:3307") \
            .option("dbtable", "users.userdemographics") \
            .option("user", "root") \
            .option("password", "example") \
            .load()

        demographic_data.show()

        demographic_data_transformed = demographic_data \
            .withColumn("country_l", lit("country")) \
            .withColumn("gender_l", lit("gender")) \
            .withColumn("state_l", lit("state")) \
            .withColumn("tags", expr("map(country_l, country, gender_l, gender, state_l, state)")) \
            .select(col("i").alias("id"), col("age"), col("tags"))

        demographic_data_transformed.show()

        # Join streams
        joined_stream = user_activity_stream_des.join(
            demographic_data_transformed,
            user_activity_stream_des.col("activity.id") == demographic_data_transformed.col("id")
        )

        joined_stream_with_tags = joined_stream.withColumn(
            "tags", expr("map_concat(activity.tags, tags)")
        )

        final_df = joined_stream_with_tags.select(
            col("activity.id").alias("id"),
            col("activity.orderid").alias("orderid"),
            col("activity.campaignid").alias("campaignid"),
            col("activity.total_amount").alias("total_amount"),
            col("activity.units").alias("units"),
            col("tags").alias("tags"),
            col("age").alias("age")
        )

        final_df.printSchema()

        # Serialize data back to Avro and write to Kafka
        output = final_df.selectExpr(
            "to_avro(struct(*)) as value"
        )

        query = output.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "http://localhost:9092") \
            .option("topic", "user_activity_demographics") \
            .option("checkpointLocation", "/tmp/") \
            .start()

        query.awaitTermination()

    except StreamingQueryException as e:
        print(f"Streaming query exception: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
