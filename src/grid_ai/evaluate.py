"""Model evaluation module."""

import os
import torch
import logging
import numpy as np
import _pickle # Import needed for exception handling
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader
# Make sure necessary classes are importable for torch.load(weights_only=False)
import torch_geometric # <-- Import added here
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool # <-- Import needed for aggregation
from grid_ai.data.schema import GridFeatures, NormalizationParams # Import if potentially used within Data objects


from . import utils
from .model import GridSecurityModel

logger = logging.getLogger(__name__)

# Register safe globals needed IF loading complex objects with weights_only=False
# This helps ensure torch.load knows about these custom/library classes.
# Note: For loading datasets/objects, weights_only=False is often required anyway.
# Let's add potentially relevant ones based on preprocessing and training.
# Check if these attributes exist before adding them, in case versions differ
safe_globals_list = [Data, GridFeatures, NormalizationParams]
if hasattr(torch_geometric.data.data, 'DataEdgeAttr'):
    safe_globals_list.append(torch_geometric.data.data.DataEdgeAttr)
if hasattr(torch_geometric.data.data, 'DataTensorAttr'):
    safe_globals_list.append(torch_geometric.data.data.DataTensorAttr)

torch.serialization.add_safe_globals(safe_globals_list)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Model predictions

    Returns:
        Dictionary of metric names and values
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Handle potential multi-output shape (N, num_outputs) vs (N,)
    # If shape is (N, 1), reshape to (N,) for sklearn metrics
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # Check if shapes are compatible after potential reshaping
    if y_true.shape != y_pred.shape:
         # Attempt multioutput='raw_values' if shapes differ fundamentally (e.g., (N, 2) vs (N,))
         # Or log an error - for now, logging error as shapes should match.
        logger.error(f"Shape mismatch in compute_metrics: y_true={y_true.shape}, y_pred={y_pred.shape}")
        # Return default/NaN values or raise error
        return { 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan }


    try:
        # Use multioutput='uniform_average' for metrics if output is multi-dimensional
        multioutput_strategy = 'uniform_average' if y_true.ndim > 1 and y_true.shape[1] > 1 else 'raw_values'

        mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput_strategy)
        mse = mean_squared_error(y_true, y_pred, multioutput=multioutput_strategy)
        r2 = r2_score(y_true, y_pred, multioutput=multioutput_strategy)

        # If uniform_average wasn't used but result is array (e.g. raw_values for multi-dim), average it
        if isinstance(mae, np.ndarray): mae = np.mean(mae)
        if isinstance(mse, np.ndarray): mse = np.mean(mse)
        if isinstance(r2, np.ndarray): r2 = np.mean(r2)


        return {
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2)
        }
    except ValueError as e:
        logger.error(f"Error computing metrics (potentially due to shape mismatch or NaNs): {e}")
        return { 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan }


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
    # Ensure results_dir exists for saving results
    results_dir = config.get('results_dir', 'results/evaluation') # Default if not in config
    os.makedirs(results_dir, exist_ok=True)
    # Also ensure log_dir exists
    log_dir = config.get('log_dir', 'logs/evaluation') # Default if not in config
    os.makedirs(log_dir, exist_ok=True)

    logger = utils.setup_logging(log_dir, 'evaluation') # Use the possibly defaulted log_dir

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    try:
        # Load checkpoint - model weights don't usually contain complex objects, so default weights_only=True is often fine
        # However, if optimizer state etc. IS saved and causes issues, might need weights_only=False here too.
        # Let's try default first for the model checkpoint.
        checkpoint = torch.load(model_path, map_location=device) # Try default first
        model = GridSecurityModel(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
    except KeyError as e:
         logger.error(f"Error loading model state_dict from checkpoint {model_path}. Key missing: {e}. Checkpoint keys: {checkpoint.keys()}")
         raise
    except _pickle.UnpicklingError as e:
         logger.error(f"Unpickling error loading model checkpoint {model_path}. Trying with weights_only=False. Error was: {e}")
         try:
             # Retry loading checkpoint with weights_only=False if default failed
             checkpoint = torch.load(model_path, map_location=device, weights_only=False)
             model = GridSecurityModel(config).to(device)
             model.load_state_dict(checkpoint['model_state_dict'])
             model.eval()
             logger.info(f"Model loaded successfully from {model_path} using weights_only=False.")
         except Exception as inner_e:
             logger.error(f"Failed to load model checkpoint even with weights_only=False: {inner_e}")
             raise
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    # Load test dataset if not provided
    if test_dataset is None:
        if 'test_dataset_file' not in config:
            logger.error("Configuration missing 'test_dataset_file' key.")
            raise KeyError("Configuration missing 'test_dataset_file'")
        test_dataset_path = config['test_dataset_file']
        logger.info(f"Loading test dataset from {test_dataset_path}...")
        try:
            # --- FIX: Load dataset with weights_only=False ---
            test_dataset = torch.load(test_dataset_path, weights_only=False)
            if not isinstance(test_dataset, list) or not test_dataset:
                 logger.warning(f"Loaded test dataset is not a non-empty list (type: {type(test_dataset)}). Evaluation might fail.")
                 # Handle empty list case if necessary
                 if not test_dataset:
                      logger.error("Test dataset is empty. Cannot evaluate.")
                      # Return empty/default results
                      return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}, np.array([]), np.array([])
            logger.info(f"Test dataset loaded successfully with {len(test_dataset)} samples.")
        except FileNotFoundError:
            logger.error(f"Test dataset file not found: {test_dataset_path}")
            raise
        except _pickle.UnpicklingError as e:
            logger.error(f"Error unpickling test dataset file {test_dataset_path}: {e}. Ensure the file contains valid PyTorch Geometric Data objects saved correctly.")
            raise
        except Exception as e:
            logger.error(f"Failed to load test dataset from {test_dataset_path}: {e}")
            raise


    # Create dataloader
    # Use batch_size from training config, or default to 32 if not present
    eval_batch_size = config.get('training', {}).get('batch_size', 32)
    # Handle potential case where dataset is empty after loading check
    if not test_dataset:
         logger.error("Cannot create DataLoader, test dataset is empty.")
         return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}, np.array([]), np.array([])

    try:
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size)
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}", exc_info=True)
        # Log details about the dataset items if possible
        if test_dataset:
             logger.error(f"First item type in dataset: {type(test_dataset[0])}")
             # Add more checks if needed (e.g., hasattr(item, 'x'))
        raise


    # Gather predictions
    y_true_list = []
    y_pred_list = []
    node_counts = [] # To verify output shape consistency if needed

    logger.info("Starting evaluation loop...")
    with torch.no_grad():
        batch_num = 0
        for batch in test_loader:
            batch_num += 1
            try:
                batch = batch.to(device)
                output = model(batch) # Model output should be graph-level [batch_size, output_dim]

                # Target y needs to be aggregated to graph level if it's node-level in the Data object
                # Assuming model does global pooling and outputs graph-level predictions
                # Let's aggregate the ground truth `batch.y` (node-level) to compare
                # If batch.y is ALREADY graph-level, this might need adjustment.
                # Check shape of batch.y
                if not hasattr(batch, 'y') or batch.y is None:
                    logger.error(f"Batch {batch_num} is missing 'y' attribute. Skipping.")
                    continue

                if batch.y.shape[0] == batch.num_nodes: # Node-level targets
                    # Aggregate using the same pooling as the model (assumed mean pooling)
                    # Ensure batch attribute exists
                    if not hasattr(batch, 'batch') or batch.batch is None:
                         logger.error(f"Batch {batch_num} missing 'batch' attribute needed for pooling node-level targets. Skipping.")
                         continue
                    y_true_agg = global_mean_pool(batch.y, batch.batch)
                elif hasattr(batch, 'num_graphs') and batch.y.shape[0] == batch.num_graphs: # Already graph-level targets
                    y_true_agg = batch.y
                else:
                    # Log details including num_graphs if available
                    num_graphs_str = f"Num graphs: {batch.num_graphs}" if hasattr(batch, 'num_graphs') else "Num graphs attribute missing"
                    logger.error(f"Unexpected shape for batch.y in batch {batch_num}: {batch.y.shape}. Num nodes: {batch.num_nodes}. {num_graphs_str}. Cannot aggregate.")
                    continue # Skip batch

                # Ensure shapes match before appending
                if output.shape != y_true_agg.shape:
                     logger.error(f"Shape mismatch in batch {batch_num}: Output shape {output.shape} != Target shape {y_true_agg.shape}. Skipping batch.")
                     continue


                y_true_list.append(y_true_agg.cpu().numpy())
                y_pred_list.append(output.cpu().numpy())
                # node_counts.append(batch.num_nodes) # Optional: track node counts

            except AttributeError as e:
                 # Catch attribute errors which might happen if Data objects are malformed
                 logger.error(f"Attribute error processing batch {batch_num}: {e}. Check Data object structure.", exc_info=True)
                 continue # Skip faulty batch
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}", exc_info=True)
                # Decide whether to skip batch or raise error
                continue # Skip faulty batch

    logger.info("Evaluation loop finished.")

    # Concatenate results if any batches were successful
    if not y_true_list or not y_pred_list:
        logger.error("No valid results collected during evaluation loop.")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}, np.array([]), np.array([])

    try:
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        logger.info(f"Concatenated results shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
    except ValueError as e:
         logger.error(f"Error concatenating results, possibly due to inconsistent shapes from batches: {e}")
         # Log shapes of individual batch results if helpful
         # for i, (yt, yp) in enumerate(zip(y_true_list, y_pred_list)):
         #      logger.debug(f"Batch {i+1} shapes: true={yt.shape}, pred={yp.shape}")
         return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}, np.array([]), np.array([])


    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred)

    # Log results
    logger.info("--- Evaluation Metrics ---")
    for name, value in metrics.items():
        logger.info(f"{name.upper()}: {value:.4f}")
    logger.info("--------------------------")

    # Save predictions and metrics
    results_save_path = os.path.join(results_dir, 'evaluation_results.pt')
    try:
        results_to_save = {
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': metrics
        }
        # Use torch.save which can handle numpy arrays
        torch.save(results_to_save, results_save_path)
        logger.info(f"Saved evaluation results (predictions and metrics) to {results_save_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")


    return metrics, y_true, y_pred

def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_dim_names: list, save_dir: str):
    """Create scatter plots of predictions vs ground truth for each output dimension.

    Args:
        y_true: Ground truth values [N, num_outputs]
        y_pred: Model predictions [N, num_outputs]
        output_dim_names: List of names for the output dimensions (e.g., ['vm_pu', 'va_degree'])
        save_dir: Directory to save plots
    """
    # Add check for empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("Cannot plot scatter: input arrays are empty.")
        return
    # Add check for correct dimensionality before accessing shape[1]
    if y_true.ndim < 2 or y_pred.ndim < 2:
         logger.error(f"Cannot plot scatter: input arrays must be 2D. Got y_true:{y_true.ndim}D, y_pred:{y_pred.ndim}D.")
         # Handle 1D case specifically if needed, e.g., plotting a single output
         if y_true.ndim == 1 and y_pred.ndim == 1 and len(output_dim_names) == 1:
              num_outputs = 1
              y_true = y_true.reshape(-1, 1) # Reshape to 2D for consistent indexing
              y_pred = y_pred.reshape(-1, 1)
         else:
              return # Exit if shapes are incompatible


    num_outputs = y_true.shape[1]
    if len(output_dim_names) != num_outputs:
        logger.warning(f"Number of output_dim_names ({len(output_dim_names)}) does not match y_true columns ({num_outputs}). Using generic names.")
        output_dim_names = [f'Output_{i}' for i in range(num_outputs)]

    os.makedirs(save_dir, exist_ok=True) # Ensure save directory exists

    for i in range(num_outputs):
        dim_name = output_dim_names[i]
        yt = y_true[:, i]
        yp = y_pred[:, i]

        # Check for NaN/inf values before plotting
        valid_mask = np.isfinite(yt) & np.isfinite(yp)
        if not np.all(valid_mask):
            logger.warning(f"NaN or inf values found in data for '{dim_name}'. Plotting only finite values.")
            yt = yt[valid_mask]
            yp = yp[valid_mask]
        if yt.size == 0: # Check if any valid points remain
             logger.warning(f"No finite data points to plot for '{dim_name}'. Skipping scatter plot.")
             continue


        plt.figure(figsize=(8, 8)) # Smaller figure size
        plt.scatter(yt, yp, alpha=0.5, s=10) # Smaller points
        # Add a diagonal line y=x for reference
        # Handle case where yt or yp might be empty after filtering
        min_val = min(np.min(yt), np.min(yp)) if yt.size > 0 else 0
        max_val = max(np.max(yt), np.max(yp)) if yt.size > 0 else 1
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        plt.xlabel(f'True {dim_name}')
        plt.ylabel(f'Predicted {dim_name}')
        plt.title(f'Prediction vs Ground Truth for {dim_name}')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Adjust axis limits slightly beyond min/max for better visualization
        range_val = max_val - min_val
        plt.xlim(min_val - 0.05 * range_val, max_val + 0.05 * range_val)
        plt.ylim(min_val - 0.05 * range_val, max_val + 0.05 * range_val)

        # plt.axis('equal') # Ensure equal aspect ratio - can sometimes make plots too small if ranges differ vastly
        plt.gca().set_aspect('equal', adjustable='box') # Better alternative for equal scaling
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{dim_name}_prediction_scatter.png')
        try:
            plt.savefig(save_path, dpi=150) # Control resolution
            logger.info(f"Saved scatter plot to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save scatter plot {save_path}: {e}")
        plt.close()


def run_diagnostics(config_path: str, model_path: str):
    """Run detailed model diagnostics.

    Args:
        config_path: Path to configuration file
        model_path: Path to saved model
    """
    # Load config
    config = utils.load_config(config_path)
    # Define plot directory from config or default
    plot_dir = config.get('plot_dir', 'plots/diagnostics') # Default if not in config
    os.makedirs(plot_dir, exist_ok=True)
    # Setup logging (might re-setup if evaluate_model also does, ensure consistency)
    log_dir = config.get('log_dir', 'logs/diagnostics') # Default if not in config
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.setup_logging(log_dir, 'diagnostics')
    logger.info("Starting model diagnostics...")

    # Run evaluation to get predictions
    try:
        metrics, y_true, y_pred = evaluate_model(config_path, model_path)
        # Check if evaluation returned valid data
        if y_true.size == 0 or y_pred.size == 0:
             logger.error("Evaluation returned empty arrays. Cannot proceed with diagnostics.")
             return
        if np.any(np.isnan(list(metrics.values()))): # Check if any metric is NaN
             logger.warning("Evaluation metrics contain NaN. Diagnostics might be unreliable.")

    except Exception as e:
        logger.error(f"Evaluation failed during diagnostics run: {e}", exc_info=True)
        return

    # Check shapes before plotting/analysis
    # Allow 1D if only one output dim, reshape within plotting function
    if y_true.shape[0] != y_pred.shape[0]: # Check only first dimension (number of samples)
        logger.error(f"Inconsistent number of samples for y_true ({y_true.shape[0]}) and y_pred ({y_pred.shape[0]}). Cannot run detailed diagnostics.")
        return
    # Check if dimensions > 2, which would be unexpected
    if y_true.ndim > 2 or y_pred.ndim > 2:
         logger.error(f"Unexpected array dimensions: y_true={y_true.ndim}D, y_pred={y_pred.ndim}D.")
         return
    # If 1D, ensure shapes match
    if y_true.ndim == 1 and y_pred.ndim == 1 and y_true.shape != y_pred.shape:
         logger.error(f"Inconsistent shapes for 1D arrays: y_true={y_true.shape}, y_pred={y_pred.shape}.")
         return
     # If 2D, ensure shapes match
    if y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape != y_pred.shape:
         logger.error(f"Inconsistent shapes for 2D arrays: y_true={y_true.shape}, y_pred={y_pred.shape}.")
         return


    # Get output dimension names from config (or default)
    # Default to handling single output if y_pred is 1D or (N,1)
    num_outputs = y_pred.shape[1] if y_pred.ndim == 2 else 1
    default_names = [f'Output_{i}' for i in range(num_outputs)]
    output_dim_names = config.get('model', {}).get('output_dim_names', default_names) # Default names


    # Create diagnostic plots
    logger.info("Generating diagnostic plots...")
    try:
        plot_prediction_scatter(
            y_true,
            y_pred,
            output_dim_names,
            plot_dir # Pass the defined plot directory
        )
    except Exception as e:
        logger.error(f"Failed to generate scatter plots: {e}", exc_info=True)


    # Analyze prediction errors
    try:
        # Reshape 1D arrays to 2D for consistent error calculation
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)

        errors = np.abs(y_true - y_pred)
        # Calculate mean absolute error per sample across outputs
        mean_sample_error = np.mean(errors, axis=1)

        # Find indices of worst predictions (highest mean absolute error)
        num_worst = min(20, len(mean_sample_error)) # Show up to 20 worst or fewer if dataset is small
        if num_worst > 0:
            # Handle potential NaNs in errors before sorting
            valid_error_indices = np.where(np.isfinite(mean_sample_error))[0]
            if len(valid_error_indices) < num_worst:
                logger.warning(f"Found only {len(valid_error_indices)} samples with finite errors. Showing fewer than {num_worst} worst predictions.")
                num_worst = len(valid_error_indices)

            if num_worst > 0:
                 # Sort based on finite errors only
                 sorted_finite_indices = np.argsort(mean_sample_error[valid_error_indices])
                 # Get the original indices corresponding to the worst finite errors
                 worst_original_idx = valid_error_indices[sorted_finite_indices[-num_worst:]][::-1]

                 logger.info(f"\n--- Top {num_worst} Worst Predictions (by Mean Absolute Error across outputs) ---")
                 for i, idx in enumerate(worst_original_idx):
                     true_vals = ", ".join([f"{val:.3f}" for val in y_true[idx]])
                     pred_vals = ", ".join([f"{val:.3f}" for val in y_pred[idx]])
                     error_vals = ", ".join([f"{val:.3f}" for val in errors[idx]])
                     logger.info(
                         f"{i+1}. Index {idx}: MAE={mean_sample_error[idx]:.4f}\n"
                         f"   True: ({true_vals})\n"
                         f"   Pred: ({pred_vals})\n"
                         f"   AbsErr: ({error_vals})"
                     )
                 logger.info("--------------------------------------------------------------------")
            else:
                 logger.info("No samples with finite errors found to determine worst predictions.")

        else:
             logger.info("Not enough samples to determine worst predictions.")

    except Exception as e:
        logger.error(f"Failed to analyze worst predictions: {e}", exc_info=True)

    logger.info("Diagnostics finished.")

