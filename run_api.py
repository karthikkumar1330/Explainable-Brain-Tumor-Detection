import argparse
import sys
import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.infrastructure.routes import router, initialize_api_models
from api.routes.auth_routes import auth_router, admin_router
from security.infrastructure.repository import SQLiteUserRepository


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments to run the REST API server."""
    parser = argparse.ArgumentParser(
        description="AuraScan AI - Brain MRI REST API Portal Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the API server on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reloading for development (reload code on edit)",
    )
    return parser.parse_args()


# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_api")

# Create FastAPI instance
app = FastAPI(
    title="AuraScan AI - Medical REST API",
    description="REST API interface to run classification, Grad-CAM attention hotspots, segmentation, and morphological reporting on brain MRI scans.",
    version="1.0.0",
)

# OWASP Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;"
    return response



@app.get("/ping")
def direct_ping():
    import datetime
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}


# Register routers
app.include_router(auth_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(router, prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    """Preloads weights, initializes security tables, and logs parameters once the server boots."""
    logger.info("Initializing security database tables and bootstrapping Admin user...")
    db_path = os.environ.get("DB_PATH", "outputs/clinical_reports.db")
    sec_repo = SQLiteUserRepository(db_path=db_path)
    sec_repo.initialize_security_tables()
    sec_repo.bootstrap_admin()

    initialize_api_models()

    # Start Email Retry Scheduler (Phase G6)
    try:
        from clinical_reporting.application.scheduler import EmailRetryScheduler
        scheduler = EmailRetryScheduler(db_path=db_path)
        scheduler.start()
        logger.info("Email Retry Scheduler started successfully on API startup.")
    except Exception as se:
        logger.error(f"Failed to start Email Retry Scheduler on API startup: {se}")


@app.on_event("shutdown")
def on_shutdown() -> None:
    """Safely shuts down background components."""
    try:
        from clinical_reporting.application.scheduler import EmailRetryScheduler
        scheduler = EmailRetryScheduler()
        scheduler.stop()
        logger.info("Email Retry Scheduler stopped successfully on API shutdown.")
    except Exception as se:
        logger.error(f"Failed to stop Email Retry Scheduler on API shutdown: {se}")


def main() -> None:
    args = parse_args()
    try:
        uvicorn.run("run_api:app", host=args.host, port=args.port, reload=args.reload)
    except Exception as e:
        print(f"Failed to start REST API server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
