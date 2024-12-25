#### setup env
```
conda create --prefix ./env
conda activate ./env
```

#### run kafka, influxdb and grafana services
```
cd dashboard
docker-compose up -d
```

#### run batch job to ingest user demographic data to mysql
```
cd dashboard/batch
python UserDemographic.py
```

#### run stream producer
```
cd dashboard/stream
python UserActivityEventProducer.py
```

#### run stream consumer for writing aggregation data by country to mysql database
```
cd dashboard/stream
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4,org.apache.spark:spark-avro_2.12:3.5.4 --jars C:\navod\Academic\sem7\bizsuit\dashboard\mysql-connector-java-8.0.20.jar SparkKafkaConsumerForConsumerIdCountry.py
```

#### run stream consumer for joining demographic data of mysql database with event data and writing to kafka
```
cd dashboard/stream
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.4,org.apache.spark:spark-avro_2.12:3.5.4 --jars C:\navod\Academic\sem7\bizsuit\dashboard\mysql-connector-java-8.0.20.jar SparkKafkaConsumerActivityNDemographic.py
```

#### 

#### run frontend:
```
cd frontend
streamlit run app.py
```