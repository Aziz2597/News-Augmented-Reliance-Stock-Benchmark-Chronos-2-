FROM python:3.11-slim
WORKDIR /app

# Copy dependencies first to leverage Docker caching
COPY requirements.txt .

# Install web server and core requirements
RUN pip install fastapi uvicorn --no-cache-dir
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY . .

# Expose port and run the API
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]