"""Automated tests validating GitHub Actions CI and Release workflows.

Ensures:
- Workflow YAML files exist and are well-formed.
- The matrix covers supported Python versions (3.9 - 3.13) and OS platforms (Linux, Windows, macOS).
- Base tests run without optional backends.
- Optional backends (MediaPipe) run in isolated environments.
- Network-dependent model tests are separate from ordinary offline unit tests.
- Trusted Publishing (OIDC) is used without hardcoded publishing tokens or credentials.
"""

from pathlib import Path
import re

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def test_workflow_files_exist() -> None:
    """Ensure both ci.yml and release.yml exist."""
    ci_yml = WORKFLOWS_DIR / "ci.yml"
    release_yml = WORKFLOWS_DIR / "release.yml"

    assert ci_yml.is_file(), "Missing .github/workflows/ci.yml"
    assert release_yml.is_file(), "Missing .github/workflows/release.yml"


def test_ci_workflow_structure_and_jobs() -> None:
    """Validate that ci.yml defines all required quality check stages."""
    content = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")

    # Required jobs
    required_jobs = [
        "lint-and-typecheck",
        "test-base",
        "test-optional-backends",
        "model-integration-tests",
        "build-and-verify-dist",
    ]
    for job in required_jobs:
        assert job in content, f"Missing required job '{job}' in ci.yml"

    # Supported Python versions in test matrix
    for py_ver in ["3.9", "3.10", "3.11", "3.12", "3.13"]:
        assert f'"{py_ver}"' in content, f"Python version {py_ver} missing from ci.yml matrix"

    # Supported operating systems
    assert "ubuntu-latest" in content
    assert "windows-latest" in content
    assert "macos-latest" in content

    # Offline unit testing in test-base
    assert '-m "not network and not optional_backend"' in content

    # Optional backend testing in test-optional-backends
    assert '-m "optional_backend"' in content
    assert "mediapipe" in content

    # Network-dependent model test separation
    assert '-m "network"' in content

    # Distribution and CLI checks
    assert "python -m build" in content
    assert "twine check --strict" in content
    assert "pip install dist/" in content
    assert "detekface --help" in content


def test_release_workflow_trusted_publishing() -> None:
    """Validate release workflow retains Trusted Publishing with OIDC token exchange."""
    content = (WORKFLOWS_DIR / "release.yml").read_text(encoding="utf-8")

    # Verify Trusted Publishing configuration
    assert "id-token: write" in content, "release.yml must specify 'id-token: write' for PyPI Trusted Publishing"
    assert "pypa/gh-action-pypi-publish" in content, "release.yml must use official gh-action-pypi-publish"
    assert "environment:" in content
    assert "name: pypi" in content

    # Verify no hardcoded credentials
    secret_patterns = [
        r"pypi-AgEI[a-zA-Z0-9_\-]+",
        r"password:\s*['\"][^$]",
        r"api_token:\s*['\"][^$]",
    ]
    for pattern in secret_patterns:
        assert not re.search(pattern, content), f"Potential hardcoded credential matched pattern: {pattern}"


def test_no_publishing_credentials_in_repository() -> None:
    """Scan all tracked configuration and python files for hardcoded PyPI credentials."""
    repo_root = Path(__file__).resolve().parent.parent
    token_prefix = "".join(["pypi-", "AgEI"])

    for f in repo_root.rglob("*.py"):
        if ".venv" in f.parts or ".git" in f.parts or "dist" in f.parts or "build" in f.parts or f.name == "test_ci_workflows.py":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        assert token_prefix not in text, f"PyPI token found in {f}"

    for f in repo_root.rglob("*.yml"):
        if ".venv" in f.parts or ".git" in f.parts:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        assert token_prefix not in text, f"PyPI token found in {f}"
