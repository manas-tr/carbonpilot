FROM python:3.11-slim

WORKDIR /app

ARG CACHE_BUST=1

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-10000}