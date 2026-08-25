FROM python:3.12-slim

WORKDIR /app

ENV PROJECT_ROOT=/app

COPY services/data-validation-service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY services/data-validation-service/src ./src

CMD ["python", "src/main.py"]