FROM dtcooper/raspberrypi-os:bookworm

ENV PIP_INDEX_URL=https://www.piwheels.org/simple \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/usr/lib/python3/dist-packages:/usr/local/lib/python3.11/dist-packages \
    PYTHONNOUSERSITE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-opencv \
    libcamera-dev \
    libcamera-apps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages -r /tmp/requirements.txt

COPY backend/ /app/

ENV CAMERA_ENABLED=1 \
    RECORD_OUTPUT_DIR=/data/recordings

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
