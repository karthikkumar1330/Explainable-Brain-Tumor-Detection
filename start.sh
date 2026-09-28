#!/usr/bin/env bash
# ==============================================================================
# AuraScan AI - Unified Production Web Service Startup Script
# Single-Service Co-located Architecture for Render
# Boots:
#   1. Internal FastAPI AI Inference Engine (127.0.0.1:8000)
#   2. Public Flask Clinical Dashboard (0.0.0.0:$PORT)
# ==============================================================================
set -eo pipefail

echo "=================================================="
echo "AuraScan AI — Production College Demo Bootstrapper"
echo "=================================================="

# 1. Create required runtime directories
echo "[1/4] Ensuring runtime storage directories exist..."
mkdir -p outputs/clinical_reports
mkdir -p outputs/temp_uploads
mkdir -p dashboard/uploads/avatars

# 2. Configure network and routing defaults
PORT="${PORT:-5000}"
FASTAPI_PORT="${FASTAPI_PORT:-8000}"
FASTAPI_HOST="127.0.0.1"

# Internal loopback configuration for Flask-to-FastAPI proxying
export FASTAPI_INTERNAL_URL="http://${FASTAPI_HOST}:${FASTAPI_PORT}"
export AURASCAN_HOST="0.0.0.0"

FASTAPI_PID=""
WATCHDOG_PID=""

# 3. Setup signal handling for graceful shutdown
cleanup() {
    echo ""
    echo "[AuraScan AI] Shutdown signal received. Cleaning up processes..."
    if [ -n "$WATCHDOG_PID" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if [ -n "$FASTAPI_PID" ] && kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "[AuraScan AI] Stopping FastAPI AI backend (PID: $FASTAPI_PID)..."
        kill -TERM "$FASTAPI_PID" 2>/dev/null || true
        wait "$FASTAPI_PID" 2>/dev/null || true
    fi
    echo "[AuraScan AI] Shutdown complete."
    exit 0
}
trap cleanup SIGTERM SIGINT EXIT

# 4. Start FastAPI AI backend in the background on loopback interface
echo "[2/4] Starting internal AI Inference REST API on ${FASTAPI_HOST}:${FASTAPI_PORT}..."
python run_api.py --host "${FASTAPI_HOST}" --port "${FASTAPI_PORT}" &
FASTAPI_PID=$!

# 5. Synchronously wait for FastAPI health endpoint (/ping)
echo "[3/4] Waiting for AI Inference REST API to become healthy..."
MAX_WAIT_SECONDS=30
WAITED=0
HEALTH_URL="http://${FASTAPI_HOST}:${FASTAPI_PORT}/ping"
HEALTHY=0

while [ $WAITED -lt $MAX_WAIT_SECONDS ]; do
    if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "[FATAL] FastAPI process terminated unexpectedly during startup."
        wait "$FASTAPI_PID" || true
        exit 1
    fi

    # Probe FastAPI /ping endpoint using Python standard library (no external curl dependency required)
    if python -c "import urllib.request; resp = urllib.request.urlopen('${HEALTH_URL}', timeout=2); sys_exit = 0 if resp.getcode() == 200 else 1; exit(sys_exit)" 2>/dev/null; then
        HEALTHY=1
        echo "[SUCCESS] AI Inference REST API is healthy and listening on ${FASTAPI_INTERNAL_URL}."
        break
    fi

    WAITED=$((WAITED + 1))
    echo "Waiting for AI REST API to boot... (${WAITED}s/${MAX_WAIT_SECONDS}s)"
    sleep 1
done

if [ $HEALTHY -ne 1 ]; then
    echo "[FATAL] AI Inference REST API failed to respond on ${HEALTH_URL} within ${MAX_WAIT_SECONDS} seconds."
    kill -TERM "$FASTAPI_PID" 2>/dev/null || true
    exit 1
fi

# 6. Start background watchdog: if FastAPI crashes during runtime, fail the service safely
(
    MAIN_PID=$$
    while kill -0 "$FASTAPI_PID" 2>/dev/null; do
        sleep 2
    done
    echo "[FATAL] AI Inference REST API (PID: $FASTAPI_PID) died unexpectedly. Terminating dashboard..."
    kill -TERM "$MAIN_PID" 2>/dev/null || true
) &
WATCHDOG_PID=$!

# 7. Start Flask Clinical Dashboard in foreground on public interface
echo "[4/4] Starting AuraScan AI Web Dashboard on 0.0.0.0:${PORT}..."

if command -v gunicorn >/dev/null 2>&1; then
    echo "[INFO] Running via Gunicorn WSGI server (1 worker, 4 threads)..."
    exec gunicorn "dashboard.infrastructure.web_server:create_app()" \
        --bind "0.0.0.0:${PORT}" \
        --workers 1 \
        --threads 4 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile -
else
    echo "[INFO] Gunicorn not found; running via Python dashboard entrypoint..."
    exec python run_dashboard.py --host 0.0.0.0 --port "${PORT}"
fi
