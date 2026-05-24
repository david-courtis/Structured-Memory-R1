# Setuptools shim. The canonical project metadata lives in pyproject.toml.
# This file exists only so legacy tooling that invokes `python setup.py ...`
# still works.
from setuptools import setup

setup()
