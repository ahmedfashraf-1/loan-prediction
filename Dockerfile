FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y gcc build-essential

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn fastapi_app.main:app --host 0.0.0.0 --port $PORT"]

