"""Visualization utilities for power grid security analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         labels: Optional[List[str]] = None,
                         title: str = 'Confusion Matrix',
                         figsize: tuple = (8, 6)) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['No Failure', 'Failure']
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    return fig

def plot_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray,
                   title: str = 'ROC Curve',
                   figsize: tuple = (8, 6)) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return fig

def plot_feature_importance(importance_scores: pd.Series,
                          title: str = 'Feature Importance',
                          figsize: tuple = (10, 6),
                          top_n: Optional[int] = None) -> plt.Figure:
    """Plot feature importance scores.
    
    Args:
        importance_scores: Series of feature importance scores
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show
        
    Returns:
        matplotlib Figure object
    """
    if top_n is not None:
        importance_scores = importance_scores.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    importance_scores.plot(kind='barh')
    
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    
    return fig

def plot_grid_metrics(results_df: pd.DataFrame,
                     numeric_cols: List[str],
                     by_failure: bool = True,
                     figsize: tuple = (15, 10)) -> plt.Figure:
    """Plot distributions of grid metrics.
    
    Args:
        results_df: DataFrame with grid metrics
        numeric_cols: List of numeric columns to plot
        by_failure: Whether to split by failure status
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if by_failure:
            sns.boxplot(data=results_df, x='failed', y=col, ax=axes[i])
            axes[i].set_xticklabels(['No Failure', 'Failure'])
        else:
            sns.histplot(data=results_df, x=col, ax=axes[i])
        
        axes[i].set_title(col)
    
    # Remove empty subplots
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def plot_failure_types(results_df: pd.DataFrame,
                      figsize: tuple = (10, 6)) -> plt.Figure:
    """Plot distribution of failure types.
    
    Args:
        results_df: DataFrame with failure information
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    failure_counts = results_df['failure_type'].value_counts()
    failure_counts.plot(kind='bar')
    
    plt.title('Distribution of Failure Types')
    plt.xlabel('Failure Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    return fig