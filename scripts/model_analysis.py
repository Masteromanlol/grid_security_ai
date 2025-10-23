import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from grid_ai.features import prepare_feature_matrix
from grid_ai.ml_pipeline import SecurityClassificationPipeline
from grid_ai.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_grid_metrics,
    plot_failure_types
)

# Set up plotting style
plt.style.use('default')  # Use default style instead
sns.set_theme()  # Set seaborn theme

def main():
    print("Loading data and model...")
    # Load data and model
    data_path = 'data/processed/case_1354_small_aggregated.pkl'
    model_path = 'models/case_1354_small_rf.pkl'

    # Load results
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

    # Load trained model
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    print(f"\nLoaded {len(df)} samples")
    print("Feature matrix shape:", df.shape)
    print("\nFailure distribution:")
    print(df['failed'].value_counts(normalize=True))

    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance = pipeline.get_feature_importance()
    plt.figure(figsize=(12, 6))
    plot_feature_importance(importance, top_n=10)
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png')
    plt.close()

    print("\nTop 10 most important features:")
    print(importance.head(10))

    # Grid metrics analysis
    print("\nAnalyzing grid metrics...")
    numeric_features = [
        'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
        'line_loading_mean', 'line_loading_std', 'line_loading_max',
        'total_p_mw', 'total_q_mvar', 'losses_mw'
    ]

    plt.figure(figsize=(15, 10))
    plot_grid_metrics(df[numeric_features + ['failed']], numeric_features)
    plt.tight_layout()
    plt.savefig('figures/grid_metrics.png')
    plt.close()

    # Failure mode analysis
    print("\nAnalyzing failure modes...")
    plt.figure(figsize=(10, 6))
    plot_failure_types(df)
    plt.tight_layout()
    plt.savefig('figures/failure_types.png')
    plt.close()

    # Calculate failure rates by component type
    failure_by_type = pd.crosstab(
        df['contingency'].apply(lambda x: x['type']),
        df['failed'],
        normalize='index'
    )

    print("\nFailure rates by component type:")
    print(failure_by_type)

    # Analyze isolated buses if available
    if 'isolated_buses' in df.columns:
        isolated_stats = df[df['has_isolated_buses']].groupby(
            df['contingency'].apply(lambda x: x['type'])
        )['isolated_buses'].agg(['count', 'mean', 'median', 'max'])
        
        print("\nIsolated buses statistics by component type:")
        print(isolated_stats)

    # Model performance analysis
    print("\nAnalyzing model performance...")
    cv_scores = pipeline.cross_validate(
        df[pipeline.feature_names],
        df['failed']
    )

    print("\nCross-validation scores:")
    for metric, scores in cv_scores.items():
        print(f"{metric:20s}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

    # Get predictions on full dataset
    y_pred = pipeline.predict(df[pipeline.feature_names])
    y_prob = pipeline.predict_proba(df[pipeline.feature_names])[:, 1]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(df['failed'], y_pred)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plot_roc_curve(df['failed'], y_prob)
    plt.tight_layout()
    plt.savefig('figures/roc_curve.png')
    plt.close()

    # Performance by component type
    print("\nCalculating performance by component type...")
    performance_by_type = []
    for comp_type in df['contingency'].apply(lambda x: x['type']).unique():
        mask = df['contingency'].apply(lambda x: x['type']) == comp_type
        metrics = pipeline.evaluate(
            df[mask][pipeline.feature_names],
            df[mask]['failed']
        )
        performance_by_type.append({
            'type': comp_type,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })

    print("\nPerformance by component type:")
    print(pd.DataFrame(performance_by_type).set_index('type').round(3))

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    main()