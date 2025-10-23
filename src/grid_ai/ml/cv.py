"""Cross-validation helpers for ML workflows."""
from __future__ import annotations

from typing import Any, Optional

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold, cross_validate


def stratified_kfold(n_splits: int = 5, shuffle: bool = True, random_state: Optional[int] = None):
    """Return a StratifiedKFold splitter configured for classification tasks."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def time_series_split(n_splits: int = 5):
    """Return a TimeSeriesSplit splitter for rolling/expanding-window evaluation."""
    return TimeSeriesSplit(n_splits=n_splits)


def group_kfold(n_splits: int = 5):
    """Return a GroupKFold splitter for grouped experiments."""
    return GroupKFold(n_splits=n_splits)


def nested_cv(estimator: Any, X, y, outer_cv, inner_cv, scoring: Optional[str] = None):
    """Run a simple nested-style evaluation using cross_validate as a placeholder.

    This function is intentionally lightweight: for full nested CV with hyperparameter search
    users should call sklearn.model_selection.GridSearchCV inside the inner loop.
    """
    # Outer CV: compute cross-validated scores (placeholder)
    res = cross_validate(estimator, X, y, cv=outer_cv, scoring=scoring, return_estimator=True)
    return res
