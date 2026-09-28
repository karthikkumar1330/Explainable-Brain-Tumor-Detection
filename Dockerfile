# 1. Base Image selection
FROM python:3.11-slim

# 2. Configure system paths and environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PORT=5000 \
    AURASCAN_HOST=0.0.0.0 \
    FASTAPI_INTERNAL_URL=http://127.0.0.1:8000

WORKDIR /app

# 3. Install operating system dependency libraries for headless OpenCV and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirement list and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5. Copy the remaining repository files
COPY . .

# 6. Ensure startup script is executable
RUN chmod +x start.sh

# 7. Expose the public Flask Dashboard port (Render routes dynamic $PORT to this container)
EXPOSE 5000

# 8. Start unified single-service co-located runtime (FastAPI internal + Flask public)
CMD ["./start.sh"]
