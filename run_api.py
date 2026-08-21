from dotenv import load_dotenv
load_dotenv()

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
from security.application.config import app_config


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments to run the REST API server."""
    parser = argparse.ArgumentParser(
        description="AuraScan AI - Brain MRI REST API Portal Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=app_config.host,
        help=f"Host interface to bind server to (default: {app_config.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=app_config.api_port,
        help=f"Port to run the API server on (default: {app_config.api_port})",
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

# Configure dynamic allowed CORS origins to support cross-device testing
cors_origins = ["http://127.0.0.1:5000", "http://localhost:5000"]
if app_config.public_base_url:
    cors_origins.append(app_config.public_base_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(cors_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Telemetry Middleware
@app.middleware("http")
async def add_request_telemetry(request: Request, call_next):
    from security.infrastructure.audit_context import audit_context
    import os
    import time
    import datetime
    from persistence.infrastructure.repository import SQLitePersistenceRepository

    trust_proxy = os.environ.get("TRUST_PROXY", "0") == "1"
    client_ip = request.client.host if request.client else "127.0.0.1"
    if trust_proxy:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            client_ip = xff.split(",")[0].strip()

    user_agent = request.headers.get("user-agent", "Unknown")

    token = audit_context.set({
        "client_ip": client_ip,
        "user_agent": user_agent
    })

    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        raise exc
    finally:
        audit_context.reset(token)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Resolve normalized route
        route_obj = request.scope.get("route")
        if route_obj:
            route_path = getattr(route_obj, "path", request.url.path)
            if request.url.path.startswith("/api") and not route_path.startswith("/api"):
                route_path = "/api" + route_path
            import re
            normalized_route = re.sub(r"\{([^}]+)\}", r"<\1>", route_path)
        else:
            normalized_route = request.url.path

        # Strip trailing slash for consistency
        if normalized_route.endswith("/") and len(normalized_route) > 1:
            normalized_route = normalized_route[:-1]

        # Exclude public health probes
        if normalized_route not in ["/ping", "/health", "/api/ping", "/api/health"]:
            try:
                db_path = os.environ.get("DB_PATH", "outputs/clinical_reports.db")
                db_repo = SQLitePersistenceRepository(db_path=db_path)
                db_repo.save_http_request_telemetry(
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    duration_ms=duration_ms,
                    method=request.method,
                    route=normalized_route,
                    status_code=status_code
                )
            except Exception as db_err:
                # Telemetry failure must not break the request
                pass


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
    logger.info("Starting AuraScan AI REST API Server...")
    logger.info(f" - API Host Bind: {args.host}")
    logger.info(f" - API Port: {args.port}")
    logger.info(f" - Allowed CORS public origin: {app_config.public_base_url}")
    try:
        uvicorn.run("run_api:app", host=args.host, port=args.port, reload=args.reload)
    except Exception as e:
        logger.error(f"Failed to start REST API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
