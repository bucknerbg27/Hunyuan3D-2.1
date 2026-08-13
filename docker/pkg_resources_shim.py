"""
Minimal pkg_resources shim for setuptools>=70 compatibility.

setuptools>=70 removed pkg_resources, but pytorch-lightning 1.9.5's
lightning_fabric still imports it for declare_namespace().
"""
import importlib.metadata
import os
import sys


def declare_namespace(packageName):
    """No-op — namespace packages are handled natively in Python 3.12."""
    pass


def resource_filename(package_or_requirement, resource_name):
    """Fallback resource lookup on sys.path."""
    for p in sys.path:
        candidate = os.path.join(p, resource_name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(resource_name)


# Expose common attributes that callers may access
__version__ = "0.0.0"
