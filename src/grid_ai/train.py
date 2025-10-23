"""Model training module."""

import os
import torch
import logging
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from . import utils
from .model import GridSecurityModel

logger = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, scheduler, device, max_grad_norm=1.0):
    """Train model for one epoch.
    
    Args:
        model: GridSecurityModel instance
        loader: PyG DataLoader
        criterion: Loss function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Average loss for the epoch or None if training failed
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    try:
        for batch in loader:
            # Move batch to device
            try:
                batch = batch.to(device)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    logger.error("CUDA out of memory. Try reducing batch size.")
                raise
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            
            # Check for NaN output
            if torch.isnan(output).any():
                logger.error("NaN values in model output")
                return None
                
            loss = criterion(output, batch.y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.error("NaN loss encountered")
                return None
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item() * batch.num_graphs
            num_batches += 1
            
            # Log batch statistics
            if num_batches % 100 == 0:
                logger.info(f"Batch {num_batches}: Loss = {loss.item():.4f}")
        
        return total_loss / len(loader.dataset)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return None

def validate_model(model, loader, criterion, device):
    """Validate model on validation set.
    
    Args:
        model: GridSecurityModel instance
        loader: PyG DataLoader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.y)
            total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)

def save_model(model, optimizer, epoch, loss, path):
    """Save model checkpoint.
    
    Args:
        model: GridSecurityModel instance
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def validate_config(config: dict) -> None:
    """Validate training configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required parameters are missing or invalid
    """
    required_keys = ['log_dir', 'processed_dataset_file', 'model_save_dir', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
            
    training_params = config['training']
    required_training_params = [
        'batch_size', 'learning_rate', 'num_epochs',
        'patience', 'val_split', 'max_grad_norm'
    ]
    for param in required_training_params:
        if param not in training_params:
            raise ValueError(f"Missing required training parameter: {param}")
            
    if not (0 < training_params['val_split'] < 1):
        raise ValueError("val_split must be between 0 and 1")
    if training_params['batch_size'] < 1:
        raise ValueError("batch_size must be positive")

def run_training_pipeline(config_path):
    """Run the complete training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Load and validate config
    config = utils.load_config(config_path)
    validate_config(config)
    logger = utils.setup_logging(config['log_dir'], 'training')
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Set memory growth
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load dataset
        dataset = torch.load(config['processed_dataset_file'])
        if len(dataset) == 0:
            raise ValueError("Empty dataset")
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Load normalization parameters
        norm_params = torch.load(config['normalization_params_file'])
        logger.info("Loaded normalization parameters")
        
        # Split dataset
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)),
            test_size=config['training']['val_split'],
            random_state=42
        )
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        
        # Create dataloaders with worker init
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            num_workers=4,
            worker_init_fn=worker_init_fn
        )
        
        # Initialize model
        model = GridSecurityModel(config).to(device)
        
        # Setup training
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        patience = config['training']['patience']
        patience_counter = 0
        max_grad_norm = config['training']['max_grad_norm']
        
        # Create checkpoint directory
        os.makedirs(config['model_save_dir'], exist_ok=True)
        
        for epoch in range(config['training']['num_epochs']):
            try:
                # Train
                train_loss = train_epoch(
                    model, train_loader, criterion, optimizer,
                    scheduler, device, max_grad_norm
                )
                
                if train_loss is None:  # Training failed
                    logger.error("Training failed, stopping...")
                    break
                
                # Validate
                val_loss = validate_model(model, val_loader, criterion, device)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['learning_rates'].append(
                    optimizer.param_groups[0]['lr']
                )
                
                # Log progress
                logger.info(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, "
                    f"LR = {optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    save_model(
                        model, optimizer, epoch, val_loss,
                        os.path.join(config['model_save_dir'], f'checkpoint_{epoch}.pt')
                    )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_model(
                        model, optimizer, epoch, val_loss,
                        os.path.join(config['model_save_dir'], 'best_model.pt')
                    )
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch} epochs")
                    break
                    
            except Exception as e:
                logger.error(f"Error during epoch {epoch}: {str(e)}")
                # Save emergency checkpoint
                save_model(
                    model, optimizer, epoch, float('inf'),
                    os.path.join(config['model_save_dir'], 'emergency_checkpoint.pt')
                )
                raise
        
        # Save training history
        torch.save(history, os.path.join(config['model_save_dir'], 'training_history.pt'))
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def train_model(config_path):
    """Main training function."""
    run_training_pipeline(config_path)