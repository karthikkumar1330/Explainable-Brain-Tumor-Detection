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
