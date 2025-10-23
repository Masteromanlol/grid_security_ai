"""Hyperparameter optimization helpers (lightweight wrappers)."""
from __future__ import annotations

from typing import Any, Dict, Optional

def grid_search_stub(estimator, param_grid: Dict[str, Any], X, y, cv: int = 3):
    """Very small wrapper that selects the first parameter combo as a placeholder."""
    # Placeholder: return estimator with params set to first combo
    keys = list(param_grid.keys())
    if not keys:
        return estimator, {}
    first_combo = {k: param_grid[k][0] for k in keys}
    try:
        estimator.set_params(**first_combo)
    except Exception:
        pass
    return estimator, first_combo
