FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app.py /app
COPY requirements.txt /app
COPY entrypoint.sh /app/entrypoint.sh
COPY service /app/service

RUN pip3 install --no-cache-dir --upgrade  pip setuptools wheel &&\
    pip3 install --no-cache-dir -r requirements.txt

RUN chmod +x /app/entrypoint.sh

ENV PORT=8766
ENV API="true"
ENV POOL_SIZE=1
ENV MODEL_TYPE="tiny.en"
ENV CUDA="true"

EXPOSE ${PORT}

ENTRYPOINT ["/app/entrypoint.sh"]