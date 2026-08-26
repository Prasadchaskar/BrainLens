FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    --break-system-packages \
    --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    Pillow \
    python-dotenv \
    mlflow

COPY services/evaluation-service/src ./src

CMD ["python", "src/main.py"]
