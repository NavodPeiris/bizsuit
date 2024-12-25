from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.sql.streaming import Trigger
from pyspark.sql.types import MapType, StringType, IntegerType, StructType
import json
import os

os.environ["PYSPARK_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"

def main():
    try:
        relative_connector_path = "../mysql-connector-java-8.0.20.jar"
        abs_connector_path = os.path.abspath(relative_connector_path)

        # Initialize Spark session
        spark = SparkSession.builder \
            .master("local") \
            .appName("realtime spark kafka consumer") \
            .config("spark.jars", abs_connector_path) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
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
        
        # Define schemas
        user_activity_schema = StructType() \
            .add("id", IntegerType()) \
            .add("campaignid", IntegerType()) \
            .add("orderid", IntegerType()) \
            .add("total_amount", IntegerType()) \
            .add("units", IntegerType()) \
            .add("tags", MapType(StringType(), StringType()))

        # Deserialize Kafka messages using the PySpark schema
        # Assuming Kafka value is JSON encoded
        df2 = input_stream.selectExpr("CAST(value AS STRING) as json_value") \
            .select(
                expr(f"from_json(json_string, '{user_activity_schema.json()}')").alias("activity")
            )

        # Load demographic data from MySQL
        demographic_data = spark.read.format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
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
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("url", "jdbc:mysql://localhost:3307") \
                .option("dbtable", "users.countryAgg") \
                .option("user", "root") \
                .option("password", "example") \
                .mode("overwrite") \
                .save()
        '''
        query2 = df4.writeStream \
            .outputMode("complete") \
            .trigger(Trigger.ProcessingTime("10 seconds")) \
            .foreachBatch(write_to_mysql) \
            .start()

        query2.awaitTermination()
        '''
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
