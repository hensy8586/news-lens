FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend (FastAPI) and Streamlit code into the image
COPY src ./src
COPY app ./app

# Railway will set PORT, but default to 8000 for local use
ENV PORT=8000
EXPOSE 8000

# Start FastAPI via uvicorn
# src/main.py must define: app = FastAPI(...)
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]