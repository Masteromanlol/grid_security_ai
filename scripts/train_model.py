#!/usr/bin/env python
# train_model.py
import os
import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from grid_ai.features import prepare_feature_matrix
from grid_ai.ml_pipeline import SecurityClassificationPipeline
from grid_ai.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_grid_metrics,
    plot_failure_types
)

def train_model(data_path, model_output_path, plots_dir=None):
    """
    Trains a machine learning model on the aggregated simulation data.

    Args:
        data_path (str): Path to the aggregated data file.
        model_output_path (str): Path to save the trained model.
        plots_dir (str): Optional directory to save plots.
    """
    # Load the data
    results = pd.read_pickle(data_path)
    
    # Get network structure information
    net_structure = {
        'n_buses': len(results[0]['bus_results']),
        'n_lines': len(results[0]['line_results']),
        'n_trafos': len(results[0]['trafo_results']),
        'edges': [(int(row['from_bus']), int(row['to_bus'])) 
                 for _, row in results[0]['line_results'].iterrows()]
    }
    
    # Prepare feature matrix
    df = prepare_feature_matrix(results, net_structure)
    
    # Define features
    numeric_features = [
        'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
        'line_loading_mean', 'line_loading_std', 'line_loading_max',
        'total_p_mw', 'total_q_mvar', 'losses_mw',
        'trafo_loading_mean', 'trafo_loading_max',
        'component_id_normalized'
    ]
    
    categorical_features = [
        'is_line', 'is_trafo', 'has_isolated_buses'
    ]
    
    # Split the data
    X = df[numeric_features + categorical_features]
    y = df['failed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train pipeline
    pipeline = SecurityClassificationPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    if metrics['auc_roc'] is not None:
        print(f"ROC AUC: {metrics['auc_roc']:.3f}")
    
    # Cross-validation scores
    cv_scores = pipeline.cross_validate(X, y)
    print("\nCross-validation Scores:")
    for metric, scores in cv_scores.items():
        print(f"{metric}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    
    # Save model
    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {model_output_path}")
    
    # Generate and save plots if plots_dir is provided
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        fig_cm = plot_confusion_matrix(y_test, y_pred)
        fig_cm.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        
        # ROC curve
        fig_roc = plot_roc_curve(y_test, y_prob)
        fig_roc.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        
        # Feature importance
        fig_imp = plot_feature_importance(pipeline.get_feature_importance(), top_n=10)
        fig_imp.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        
        # Grid metrics
        fig_metrics = plot_grid_metrics(df, numeric_features)
        fig_metrics.savefig(os.path.join(plots_dir, 'grid_metrics.png'))
        
        # Failure types
        fig_failures = plot_failure_types(df)
        fig_failures.savefig(os.path.join(plots_dir, 'failure_types.png'))
        
        plt.close('all')
        print(f"\nPlots saved to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on simulation data.")
    parser.add_argument("data_path", type=str, help="Path to the aggregated data file.")
    parser.add_argument("model_output_path", type=str, help="Path to save the trained model.")
    parser.add_argument("--plots-dir", type=str, help="Directory to save plots (optional)")
    args = parser.parse_args()

    train_model(args.data_path, args.model_output_path, args.plots_dir)
