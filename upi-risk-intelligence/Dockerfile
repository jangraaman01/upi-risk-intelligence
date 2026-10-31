FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# bake data + trained model into the image so the container starts ready to serve (small = ~1 min; use --build-arg SCALE=medium for the full world)
ARG SCALE=small
RUN python -m src.pipeline --scale ${SCALE}
EXPOSE 8000 8501
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
