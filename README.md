#### setup env
conda create --prefix ./env
conda activate ./env

#### run kafka, influxdb and grafana services
```
cd dashboard
docker-compose up -d
```

#### run frontend:
```
cd frontend
streamlit run app.py
```