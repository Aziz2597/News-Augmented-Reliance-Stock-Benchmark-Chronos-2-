FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Git for Chronos GitHub dependency
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install PyTorch CPU first
RUN python -m pip install \
    torch==2.15.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies and web framework cleanly
RUN python -m pip install -r requirements.txt && \
    python -m pip install fastapi uvicorn && \
    python -m pip check

# Copy project files
COPY . .

# Render supplies PORT dynamically
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]