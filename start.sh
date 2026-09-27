#!/usr/bin/env bash
# ==============================================================================
# AuraScan AI - Unified Render Service Startup Script
# Boots FastAPI AI Backend (internal 127.0.0.1:8000) and Flask Dashboard (public 0.0.0.0:$PORT)
# ==============================================================================
set -e

# 1. Ensure required runtime storage directories exist
mkdir -p outputs/clinical_reports
mkdir -p outputs/temp_uploads
mkdir -p dashboard/uploads/avatars

# 2. Configure defaults for co-located container routing
export FASTAPI_URL="${FASTAPI_URL:-http://127.0.0.1:8000}"
export AURASCAN_HOST="0.0.0.0"
PORT="${PORT:-5000}"

# 3. Setup cleanup trap for graceful container shutdown
FASTAPI_PID=""
cleanup() {
    echo "[AuraScan AI] Shutdown signal received. Terminating processes..."
    if [ -n "$FASTAPI_PID" ]; then
        kill -TERM "$FASTAPI_PID" 2>/dev/null || true
        wait "$FASTAPI_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 4. Start FastAPI AI Inference Engine in the background on internal loopback
echo "[AuraScan AI] Starting FastAPI AI backend in background on 127.0.0.1:8000..."
python run_api.py --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

# 5. Non-blocking background health check logger
(
    for i in $(seq 1 60); do
        if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ping', timeout=1)" >/dev/null 2>&1; then
            echo "[AuraScan AI] FastAPI AI backend is ready and responding on /ping."
            break
        fi
        sleep 2
    done
) &

# 6. Start Flask Clinical Dashboard immediately in the foreground on public port
echo "[AuraScan AI] Starting Flask Clinical Dashboard on 0.0.0.0:${PORT}..."
python run_dashboard.py --host 0.0.0.0 --port "$PORT"
