FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to maximize Docker layer cache reuse.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy runtime source and artifacts.
COPY data_pipeline /app/data_pipeline
COPY inference/inference_service.py /app/inference_service.py
COPY data_pipeline/training_stats.npy /app/training_stats.npy
COPY training/models/model_int8.tflite /app/model.tflite

ENV MODEL_PATH=/app/model.tflite
ENV TRAINING_STATS=/app/training_stats.npy
ENV MQTT_BROKER_HOST=localhost
ENV MQTT_BROKER_PORT=1883
ENV TRUCK_ID=TRUCK_001

# Inference service reads sensors from MQTT and publishes predictions/alerts.
CMD ["python", "/app/inference_service.py"]
