FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY services/monitoring-service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY services/monitoring-service/src ./src

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]