import os
import pytest
@pytest.fixture(autouse=True)
def manage_auto_assign_env(request):
    """Autouse fixture to set/unset DISABLE_TEST_AUTO_ASSIGN based on the test module name to prevent environment leakage."""
    module_name = request.module.__name__

    # Check if the test module requires doctor auto-assignment to be disabled
    should_disable = (
        "test_h3" in module_name or
        "test_h4" in module_name or
        "test_h5" in module_name or
        "test_mri_annotations" in module_name or
        "test_clinician_notes" in module_name
    )

    original_val = os.environ.get("DISABLE_TEST_AUTO_ASSIGN")

    if should_disable:
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
    else:
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]

    yield

    # Restore environment variable after test
    if original_val is not None:
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = original_val
    else:
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]


@pytest.fixture(autouse=True)
def clean_db_path_env(request):
    """Autouse fixture to clean leaked DB_PATH pointing to production clinical_reports.db,
    and dynamically propagate self.db_path to os.environ['DB_PATH'] and route configurations."""
    instance = request.instance
    if instance and hasattr(instance, "setUp"):
        original_setup = instance.setUp

        def wrapped_setup(*args, **kwargs):
            original_setup(*args, **kwargs)
            db_path = getattr(instance, "db_path", None)
            if db_path:
                os.environ["DB_PATH"] = db_path
                try:
                    from api.infrastructure import routes as api_routes
                    from api.routes import auth_routes
                    api_routes.DEFAULT_DB_PATH = db_path
                    auth_routes.DEFAULT_DB_PATH = db_path
                except ImportError:
                    pass

        instance.setUp = wrapped_setup

    original_db_path = os.environ.get("DB_PATH")
    if original_db_path and "clinical_reports.db" in original_db_path:
        del os.environ["DB_PATH"]

    yield

    if original_db_path and "clinical_reports.db" in original_db_path:
        os.environ["DB_PATH"] = original_db_path
