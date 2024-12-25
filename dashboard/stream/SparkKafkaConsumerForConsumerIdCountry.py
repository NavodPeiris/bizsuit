from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, MapType
from pyspark.sql.avro.functions import from_avro
from pyspark.streaming import StreamingContext
import json
import os
import io
import fastavro
import pyspark.sql.functions as psf

def deserialize_avro(serialized_msg):
    bytes_io = io.BytesIO(serialized_msg)
    bytes_io.seek(0)
    avro_schema = {
        "name":"ConsumerActivity",
        "type":"record",
        "fields":[
            {
                "name":"id",
                "type":"int"
            },
            {
                "name":"campaignid",
                "type":"int"
            },
            {
                "name":"orderid",
                "type":"int"
            },
            {
                "name":"total_amount",
                "type":"int"
            },
            {
                "name":"units",
                "type":"int"
            },
            {"name":"tags","type":{"type":"map","values":"string"}}
        ]
    }

    deserialized_msg = fastavro.schemaless_reader(bytes_io, avro_schema)

    return (
        deserialized_msg["id"],
        deserialized_msg["campaignid"],
        deserialized_msg["orderid"],
        deserialized_msg["total_amount"],
        deserialized_msg["units"],
        deserialized_msg["tags"]
    )

os.environ["PYSPARK_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"

def main():
    try:
        relative_connector_path = "../mysql-connector-java-8.0.20.jar"
        abs_connector_path = os.path.abspath(relative_connector_path)

        # Initialize Spark session
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("realtime spark kafka consumer") \
            .config("spark.jars", abs_connector_path) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "8g") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4,org.apache.spark:spark-avro_2.12:3.5.4") \
            .getOrCreate()
        
        # Define Kafka source
        input_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "http://localhost:9092") \
            .option("subscribe", "consumer_activity") \
            .option("group.id", "user_activity_grp") \
            .option("schema.registry.url", "http://localhost:8081") \
            .load()
        
        df_schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("campaignid", IntegerType(), True),
            StructField("orderid", IntegerType(), True),
            StructField("total_amount", IntegerType(), True),
            StructField("units", IntegerType(), True),
            StructField("tags", MapType(StringType(), StringType()), True)
        ])

        avro_deserialize_udf = psf.udf(deserialize_avro, returnType=df_schema)
        df2 = input_stream.withColumn("avro", avro_deserialize_udf(psf.col("value"))).select("avro.*")

        # Load demographic data from MySQL
        demographic_data = spark.read.format("jdbc") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("url", "jdbc:mysql://localhost:3307") \
            .option("dbtable", "users.userdemographics") \
            .option("user", "root") \
            .option("password", "example") \
            .load()

        demographic_data.show()
        '''
        # Join Kafka stream with MySQL data
        df3 = df2.join(
            demographic_data,
            df2["id"] == demographic_data["user_id"]
        )

        # Group by country and count
        df4 = df3.groupBy("country").count()
        '''
        # Write to console
        query2 = df2.writeStream \
            .outputMode("append") \
            .format("console") \
            .start()

        query2.awaitTermination()
        
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