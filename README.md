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

#### run recommendation API:
```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```