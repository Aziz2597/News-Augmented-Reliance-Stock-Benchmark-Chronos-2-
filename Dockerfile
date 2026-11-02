
FROM python:3.12-slim

WORKDIR /app

# Python environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install CPU-only PyTorch
# Verify that this version has a Python 3.12 compatible wheel
RUN python -m pip install \
    torch==2.15.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
RUN python -m pip install -r requirements.txt

# Install API dependencies
RUN python -m pip install fastapi uvicorn

# Verify installed dependencies
RUN python -m pip check

# Verify critical imports
RUN python -c "import torch, numpy, pandas, networkx, transformers; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__)"

# Copy application files
COPY . .

# Create output directory
RUN mkdir -p /app/outputs

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
