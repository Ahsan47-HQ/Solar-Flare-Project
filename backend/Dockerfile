FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PORT=8000
WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd -r -u 10001 helion
COPY __init__.py features.py server.py /app/backend/
COPY assets/bundle.zip /tmp/bundle.zip
RUN python -c "import zipfile; zipfile.ZipFile('/tmp/bundle.zip').extractall('/app/backend/data')" && rm /tmp/bundle.zip
RUN chown -R helion:helion /app
USER helion
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn backend.server:app --host 0.0.0.0 --port ${PORT}"]
