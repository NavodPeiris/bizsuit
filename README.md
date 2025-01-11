#### setup env
```
cd frontend
pip install -r requirements.txt
```

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