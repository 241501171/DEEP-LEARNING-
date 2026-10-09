FROM python:3.11-slim
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app.py features.py ./
COPY static ./static
COPY model/aqi_model.joblib ./model/aqi_model.joblib

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
