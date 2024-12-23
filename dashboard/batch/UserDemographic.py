from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import random

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

spark = SparkSession.builder \
    .appName("Batch User Demographic Data") \
    .master("local") \
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

user_df.write \
    .format("jdbc") \
    .option("url", "jdbc:mysql://localhost:3307") \
    .option("dbtable", "users.userdemographics") \
    .option("user", "root") \
    .option("password", "example") \
    .mode("overwrite") \
    .save()

print("Data Creation is completed")
