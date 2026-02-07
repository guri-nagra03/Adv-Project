# CardioGuard Dockerfile
# Python 3.13 slim image for production deployment

FROM python:3.13-slim

# Set metadata
LABEL maintainer="CardioGuard Team"
LABEL description="Educational cardiovascular wellness monitoring system with FHIR integration"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/raw \
             /app/data/processed \
             /app/data/cache \
             /app/models \
             /app/logs

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY ui/ ./ui/
COPY scripts/ ./scripts/

# Copy .env.example as template (users should provide their own .env)
COPY .env.example ./.env.example

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command (run Streamlit app)
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
