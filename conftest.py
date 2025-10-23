"""Pytest configuration helpers.

Ensure the repository 'src' directory is on sys.path so tests can import the
`grid_ai` package during collection.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest


def pytest_sessionstart(session):
    # Insert project's src directory at the front of sys.path to allow imports
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    # Detect availability of heavy optional dependencies.
    # For 'torch' we avoid importing (it can crash on Windows with bad binaries)
    # and use find_spec. For sklearn/pandapower/lxml we attempt a safe import
    # inside try/except to detect binary compatibility issues.
    import importlib.util

    available = {}

    # torch: only check spec (do not import)
    try:
        available["torch"] = importlib.util.find_spec("torch") is not None
    except Exception:
        available["torch"] = False

    # sklearn, pandapower, lxml: try import and mark unavailable on any Exception
    for mod in ("sklearn", "pandapower", "lxml"):
        try:
            __import__(mod)
            available[mod] = True
        except Exception:
            available[mod] = False

    # Store on the session config for access in hooks
    session.config._available_deps = available


def pytest_collection_modifyitems(session, config, items):
    """Skip tests that require unavailable heavy dependencies.

    This is a conservative safeguard so test collection doesn't fail due to
    binary mismatches or missing system libraries. It uses filename heuristics
    to decide which tests to skip.
    """
    available = getattr(config, "_available_deps", {})
    skip_reasons = []

    for item in list(items):
        name = item.fspath.basename

        # Tests that import torch
        if name in ("test_model.py", "test_preprocessing.py") and not available.get("torch", False):
            reason = "skipping test: 'torch' not available or incompatible in current environment"
            item.add_marker(pytest.mark.skip(reason=reason))
            skip_reasons.append((name, reason))

        # Tests that use pandapower / lxml
        if name in ("test_simulation.py",) and not (available.get("pandapower", False) and available.get("lxml", False)):
            reason = "skipping test: 'pandapower' or 'lxml' not available"
            item.add_marker(pytest.mark.skip(reason=reason))
            skip_reasons.append((name, reason))

        # Tests that require sklearn
        if name in ("test_model_selection.py",) and not available.get("sklearn", False):
            reason = "skipping test: 'sklearn' not available or binary-incompatible"
            item.add_marker(pytest.mark.skip(reason=reason))
            skip_reasons.append((name, reason))

    # Optionally log skipped tests for visibility
    if skip_reasons:
        print("pytest: skipped tests due to missing deps:")
        for n, r in skip_reasons:
            print(f" - {n}: {r}")
