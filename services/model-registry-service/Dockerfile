FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    mlflow==3.15.1 \
    python-dotenv==1.2.2 \
    numpy==2.5.2

COPY services/model-registry-service/src ./src

CMD ["python", "src/register_model.py"]