"""Tests for package importability, versioning, and namespace integrity."""

import detekface


def test_import_package() -> None:
    """Verify that detekface can be imported cleanly."""
    assert detekface is not None


def test_package_version() -> None:
    """Verify that detekface.__version__ is defined as a non-empty semantic version string."""
    assert hasattr(detekface, "__version__")
    assert isinstance(detekface.__version__, str)
    assert len(detekface.__version__) > 0
    # Basic semver component check (e.g. 0.1.0)
    parts = detekface.__version__.split(".")
    assert len(parts) >= 2


def test_package_all_export() -> None:
    """Verify that __all__ is properly exposed and all symbols exist."""
    assert hasattr(detekface, "__all__")
    assert "__version__" in detekface.__all__
    for symbol in detekface.__all__:
        assert hasattr(detekface, symbol), f"Exported symbol '{symbol}' is missing from detekface"
