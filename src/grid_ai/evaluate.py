"""Model evaluation module."""

import os
import torch
import logging
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from . import utils
from .model import GridSecurityModel

logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def evaluate_model(config_path: str, model_path: str, test_dataset=None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate trained model on test set.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model checkpoint
        test_dataset: Optional test dataset (if None, will load from config)
        
    Returns:
        Tuple of (metrics dictionary, true values, predicted values)
    """
    # Load config and setup logging
    config = utils.load_config(config_path)
    logger = utils.setup_logging(config['log_dir'], 'evaluation')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = GridSecurityModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset if not provided
    if test_dataset is None:
        test_dataset = torch.load(config['test_dataset_file'])
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Gather predictions
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            y_true_list.append(batch.y.cpu().numpy())
            y_pred_list.append(output.cpu().numpy())
    
    # Concatenate results
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Log results
    for name, value in metrics.items():
        logger.info(f"{name.upper()}: {value:.4f}")
    
    # Save predictions
    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }
    torch.save(results, os.path.join(config['results_dir'], 'evaluation_results.pt'))
    
    return metrics, y_true, y_pred

def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Create scatter plot of predictions vs ground truth.
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs Ground Truth')
    plt.savefig(save_path)
    plt.close()

def run_diagnostics(config_path: str, model_path: str):
    """Run detailed model diagnostics.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
    """
    # Load config and setup logging
    config = utils.load_config(config_path)
    logger = utils.setup_logging(config['log_dir'], 'diagnostics')
    
    # Run evaluation
    metrics, y_true, y_pred = evaluate_model(config_path, model_path)
    
    # Create diagnostic plots
    os.makedirs(config['plot_dir'], exist_ok=True)
    
    # Voltage magnitude predictions
    plot_prediction_scatter(
        y_true[:, 0],  # vm_pu
        y_pred[:, 0],
        os.path.join(config['plot_dir'], 'voltage_magnitude_predictions.png')
    )
    
    # Voltage angle predictions
    plot_prediction_scatter(
        y_true[:, 1],  # va_degree
        y_pred[:, 1],
        os.path.join(config['plot_dir'], 'voltage_angle_predictions.png')
    )
    
    # Find worst predictions
    errors = np.abs(y_true - y_pred)
    worst_idx = np.argsort(errors.mean(axis=1))[-20:]
    
    logger.info("\nWorst 20 Predictions:")
    for idx in worst_idx:
        logger.info(
            f"Index {idx}: "
            f"True (V={y_true[idx,0]:.3f}, θ={y_true[idx,1]:.3f}) "
            f"Pred (V={y_pred[idx,0]:.3f}, θ={y_pred[idx,1]:.3f})"
        )