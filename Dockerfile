FROM python:3.11-slim

# Build version: 6.0 - Slim image without heavy NLP deps (Railway 4GB limit)
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLP model download removed - image exceeded 4GB limit
# The app works without spacy, just disables semantic/entity analysis

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
