"""Model selection helpers: compare multiple estimators using cross validation."""
from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np


def _fallback_cv_score(estimator, X, y, cv: int, rng=None):
    """A tiny fallback cross-validation implementation that doesn't rely on sklearn.

    It requires the estimator to implement fit(X, y) and either score(X, y) or predict(X).
    For predict-only estimators, accuracy is used for classification tasks.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    n = X.shape[0]
    if cv <= 1 or cv >= n:
        # single split: train/test 80/20
        idx = np.arange(n)
        rng.shuffle(idx)
        split = int(n * 0.8)
        train_idx, test_idx = idx[:split], idx[split:]
        folds = [(train_idx, test_idx)]
    else:
        # simple randomized K folds
        idx = np.arange(n)
        folds = []
        for i in range(cv):
            rng.shuffle(idx)
            split = int(n * 0.8)
            folds.append((idx[:split].copy(), idx[split:].copy()))

    scores = []
    for train_idx, test_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        try:
            est = estimator
            est.fit(X_train, y_train)
        except Exception:
            # Try fresh clone by re-instantiating if possible
            try:
                est = estimator.__class__()
                est.set_params(**getattr(estimator, "get_params", lambda: {})())
                est.fit(X_train, y_train)
            except Exception:
                raise

        if hasattr(est, "score"):
            sc = est.score(X_test, y_test)
        else:
            # fallback to accuracy for classification-like targets
            y_pred = est.predict(X_test)
            sc = float((y_pred == y_test).mean())
        scores.append(float(sc))
    return np.array(scores)


def compare_models(estimators: List[Any], X, y, cv: int = 5, scoring: Optional[str] = None) -> List[Dict[str, Any]]:
    """Evaluate a list of (name, estimator) tuples and return ranked results.

    Args:
        estimators: Iterable of (name, estimator) pairs
        X: feature matrix
        y: target vector
        cv: number of cross-validation folds
        scoring: scoring string passed to sklearn

    Returns:
        List of dicts with keys: name, mean_score, std_score, scores
    """
    results = []
    # Try to use sklearn's cross_val_score if available and importable
    try:
        from sklearn.model_selection import cross_val_score as _sk_cv

        for name, est in estimators:
            try:
                scores = _sk_cv(est, X, y, cv=cv, scoring=scoring)
            except Exception:
                # Fall back to internal implementation on any sklearn error
                scores = _fallback_cv_score(est, np.asarray(X), np.asarray(y), cv)
            results.append({
                "name": name,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "scores": np.asarray(scores).tolist(),
            })
    except Exception:
        # sklearn not importable or broken: use fallback implementation
        for name, est in estimators:
            scores = _fallback_cv_score(est, np.asarray(X), np.asarray(y), cv)
            results.append({
                "name": name,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "scores": np.asarray(scores).tolist(),
            })

    # Sort by mean_score descending
    results = sorted(results, key=lambda r: r["mean_score"], reverse=True)
    return results
