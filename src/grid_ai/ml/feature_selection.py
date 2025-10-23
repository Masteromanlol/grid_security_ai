"""Feature selection utilities: correlation, RFE, LASSO and mutual information stubs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def correlation_matrix(X) -> Dict[str, Any]:
    """Return a simple Pearson correlation matrix as a numpy array."""
    if hasattr(X, "corr"):
        # pandas DataFrame
        return X.corr()
    else:
        return np.corrcoef(X, rowvar=False)


def rfe_stub(estimator, X, y, n_features_to_select: Optional[int] = None):
    """Placeholder RFE: selects the first n features."""
    n = n_features_to_select or min(10, X.shape[1])
    return list(range(n))


def lasso_select_stub(X, y, alpha: float = 0.01):
    """Placeholder LASSO selection returning top coefficients by absolute value."""
    coefs = np.random.RandomState(0).rand(X.shape[1])
    idx = np.argsort(np.abs(coefs))[::-1]
    return idx.tolist()
