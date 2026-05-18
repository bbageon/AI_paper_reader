FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps required by pymupdf (mupdf bindings ship as wheels, but having
# build essentials handy avoids surprises on older base images)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py ./
COPY templates ./templates

# Persistent library lives here; mounted as a volume by docker-compose
RUN mkdir -p /app/data/library/Default

EXPOSE 8181

# Single-process is fine for local PoC. Streaming SSE works cleanly with gunicorn's
# gthread worker; bumped timeout so long Ollama generations don't get killed.
CMD ["gunicorn", "--bind", "0.0.0.0:8181", \
     "--workers", "1", "--threads", "8", "--worker-class", "gthread", \
     "--timeout", "600", "app:app"]
