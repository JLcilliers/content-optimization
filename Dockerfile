FROM python:3.11-slim

# Cache bust: v4 - API response formatting fix
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy language model
RUN python -m spacy download en_core_web_sm

# Copy the source code
COPY src /app/src

# Copy the backend application
COPY backend /app/backend

# Set Python path - /app so 'backend' and 'src' packages are accessible
ENV PYTHONPATH=/app:/app/src

# Default port (Railway will override with PORT env var)
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application - use shell form to expand $PORT
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
