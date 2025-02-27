FROM nvidia/cuda:12.6.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget libffi-dev liblzma-dev xz-utils tk-dev make && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz && \
    tar -xvf Python-3.12.5.tgz && \
    cd Python-3.12.5 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.12.5* /var/lib/apt/lists/*

FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=builder /usr/local /usr/local

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

WORKDIR /app

RUN python3.12 -m pip install --no-cache-dir torch torchvision torchaudio

COPY ./requirements.txt /app/requirements.txt
COPY ./config.yaml /app/config.yaml
COPY ./.env /app/.env
COPY ./src /app/src

RUN python3.12 -m pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY ./model_inference/realtime/app.py /app/app.py

ENV PYTHONPATH=/app

ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "app:app"]

