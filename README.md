#### setup env
```
pip install -r requirements.txt
```

#### run kafka, influxdb and grafana services
```
cd dashboard
docker-compose up -d
```

#### run batch job to ingest user demographic data to mysql
```

```

#### run stream producer
```

```

#### run stream consumer for writing aggregation data by country to mysql database
```

```

#### run stream consumer for joining demographic data of mysql database with event data and writing to kafka
```

```

#### run frontend:
```
cd frontend
streamlit run app.py
```