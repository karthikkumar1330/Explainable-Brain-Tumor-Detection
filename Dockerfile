# 1. Base Image selection
FROM python:3.11-slim

# 2. Configure system paths and environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# 3. Install operating system dependency libraries for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirement list and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5. Copy the remaining repository files
COPY . .

# 6. Expose the API, Streamlit, and Flask Dashboard ports
EXPOSE 8000 8501 5000

# 7. Set the default entry point command (Streamlit App)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
