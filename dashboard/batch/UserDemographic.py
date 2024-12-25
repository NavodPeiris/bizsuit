from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import random
import os

os.environ["PYSPARK_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\navod\\Academic\\sem7\\bizsuit\\env\\python.exe"

class UserDemoGraphicData:
    def __init__(self, user_id, age, gender, state, country):
        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.state = state
        self.country = country

def generate_user_data():
    countries = ["USA", "UK", "Canada", "Australia", "Germany"]
    users = []
    for i in range(100000):
        country = random.choice(countries)
        user = UserDemoGraphicData(
            user_id=i,
            age=random.randint(18, 65),
            gender=random.choice(["Male", "Female", "Other"]),
            state="State_" + str(random.randint(1, 50)),
            country=country
        )
        users.append(user)
    return users

relative_connector_path = "../mysql-connector-java-8.0.20.jar"
abs_connector_path = os.path.abspath(relative_connector_path)

spark = SparkSession.builder \
    .appName("Batch User Demographic Data") \
    .master("local[*]") \
    .config("spark.jars", abs_connector_path) \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

users = generate_user_data()
schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("age", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("state", StringType(), True),
    StructField("country", StringType(), True)
])
user_df = spark.createDataFrame(users, schema)

user_df.limit(1000).write \
    .format("jdbc") \
    .option("driver", "com.mysql.cj.jdbc.Driver") \
    .option("url", "jdbc:mysql://localhost:3307/users") \
    .option("dbtable", "users.userdemographics") \
    .option("user", "root") \
    .option("password", "example") \
    .mode("overwrite") \
    .save()

print("Data Creation is completed")
