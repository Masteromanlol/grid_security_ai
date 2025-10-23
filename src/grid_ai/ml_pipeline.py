"""Machine learning pipeline for power grid security analysis."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix
)

class SecurityClassificationPipeline:
    """Pipeline for training and evaluating security classification models."""
    
    def __init__(self, 
                 numeric_features: List[str],
                 categorical_features: List[str],
                 target_col: str = 'failed',
                 n_folds: int = 5,
                 random_state: int = 42):
        """Initialize the pipeline.
        
        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_col: Name of the target column
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Initialize cross-validation splitter
        self.cv = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Create the preprocessing pipeline
        numeric_transformer = StandardScaler()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        # Create the full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state
            ))
        ])
        
        self.feature_names = numeric_features + categorical_features
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the pipeline on training data.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        self.pipeline.fit(X[self.feature_names], y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        return self.pipeline.predict(X[self.feature_names])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities
        """
        return self.pipeline.predict_proba(X[self.feature_names])
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary of performance metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        
        try:
            auc_roc = roc_auc_score(y, self.predict_proba(X)[:, 1])
        except:
            auc_roc = None
            
        conf_matrix = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': conf_matrix
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary of cross-validation scores
        """
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = cross_validate(
            self.pipeline,
            X[self.feature_names],
            y,
            cv=self.cv,
            scoring=scoring,
            return_train_score=True
        )
        
        return {
            metric: cv_results[f'test_{metric}'].tolist()
            for metric in scoring.keys()
        }
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores.
        
        Returns:
            Series of feature importance scores
        """
        importance = self.pipeline.named_steps['classifier'].feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)