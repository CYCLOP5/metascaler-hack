FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      coinor-cbc \
      coinor-libcbc-dev \
      git \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
