FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && pip install --no-cache-dir \
        numpy \
        opencv-python-headless \
        requests \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY record_stream.py wait_and_record.py /app/

ENTRYPOINT ["python", "/app/wait_and_record.py"]
