"""Lightweight model registry for storing model metadata and artifacts.

This registry stores metadata in a JSON index and saves model files using joblib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import joblib
except Exception:
    joblib = None
import pickle
import time
import hashlib
import subprocess
import shutil


class ModelRegistry:
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path or Path.cwd() / "ml_registry")
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.repo_path / "index.json"
        if not self.index_file.exists():
            with open(self.index_file, "w") as f:
                json.dump([], f)

    def register(self, model: Any, name: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save a model and record metadata.

        Returns the path to the saved model file.
        """
        metadata = metadata or {}
        model_id = f"{name}_{len(self._read_index()) + 1}"
        model_path = self.repo_path / f"{model_id}.joblib"
        # Save model using joblib when available, otherwise pickle
        if joblib is not None:
            try:
                joblib.dump(model, model_path)
            except Exception:
                # Fallback to pickle if joblib.dump fails
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        entry = {
            "id": model_id,
            "name": name,
            "path": str(model_path.absolute()),
            "metadata": metadata,
        }

        # Attach minimal provenance metadata
        prov = {
            "saved_at": time.time(),
            "platform": subprocess.check_output(["python", "-V"]).decode().strip() if shutil.which("python") else None,
        }
        metadata = {**metadata, **prov} if metadata else prov

        index = self._read_index()
        index.append(entry)
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

        return model_path

    def list(self) -> Dict[str, Any]:
        return self._read_index()

    def load(self, model_id: str) -> Any:
        index = self._read_index()
        for entry in index:
            if entry["id"] == model_id:
                if joblib is not None:
                    return joblib.load(entry["path"])
                else:
                    with open(entry["path"], "rb") as f:
                        return pickle.load(f)
        raise KeyError(f"Model id {model_id} not found in registry")

    def _read_index(self):
        with open(self.index_file, "r") as f:
            return json.load(f)
