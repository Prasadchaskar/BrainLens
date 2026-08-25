FROM python:3.12-slim

WORKDIR /app

COPY services/data-download-service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .git ./.git
COPY .dvc ./.dvc
COPY .dvcignore ./
COPY data/raw/dataset.dvc ./data/raw/dataset.dvc

COPY services/data-download-service/src ./src

CMD ["python", "src/main.py"]