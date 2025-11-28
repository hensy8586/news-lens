FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps if needed (e.g. gcc, for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better build cache
COPY ../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . /app/backend

# Expose port (not strictly required, but good practice)
EXPOSE 8000

# Default command: uvicorn
# Most hosts set PORT env; fall back to 8000
ENV PORT=8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
