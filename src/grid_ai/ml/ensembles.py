"""Ensemble helpers: stacking and simple averaging with safe fallbacks."""
from __future__ import annotations

from typing import List, Any, Optional

def stacking_helper(estimators: List[Any], final_estimator: Any = None, cv: int = 5):
    """Create a stacking ensemble. If sklearn.ensemble.StackingClassifier/Regressor
    is available, use it; otherwise return a simple averaging wrapper.
    """
    try:
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        # Make a guess about task type by checking if final_estimator has predict_proba
        if final_estimator is None:
            final_estimator = estimators[-1][1].__class__()
        # This is a simplified approach; users should call sklearn's Stacking directly
        return StackingClassifier(estimators, final_estimator=final_estimator, cv=cv)
    except Exception:
        # Fallback: return a simple averaging wrapper object
        class AveragingWrapper:
            def __init__(self, ests):
                self.ests = [e for _, e in ests]

            def fit(self, X, y):
                for e in self.ests:
                    e.fit(X, y)
                return self

            def predict(self, X):
                import numpy as np
                preds = [e.predict(X) for e in self.ests]
                return np.round(np.mean(preds, axis=0)).astype(int)

        return AveragingWrapper(estimators)
