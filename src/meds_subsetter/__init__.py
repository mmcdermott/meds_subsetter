"""Nested, hash-based subject subsetting, resharding, sizing, and fingerprinting of MEDS datasets."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - depends on install method
    __version__ = version("meds-subsetter")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__package_name__ = "meds_subsetter"

__all__ = ["__package_name__", "__version__"]
