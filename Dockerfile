FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy server code, models, and data
COPY server/ ./server/
COPY 4_inputs_BT_only/ ./4_inputs_BT_only/
COPY insat_data/ ./insat_data/

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run WSGI server on 0.0.0.0:7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "server.app:app"]
