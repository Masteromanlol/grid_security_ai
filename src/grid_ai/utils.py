"""Utility functions for the grid_ai package."""

import os
import yaml
import random
import logging
import importlib
import numpy as np
import torch
import pandapower as pp

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir, name):
    """Set up logging configuration with file output.
    
    Args:
        log_dir: Directory to store log files
        name: Name of the logger/log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed for random number generators.
        deterministic_cudnn: If True, make CUDA operations deterministic.
            This may impact performance but ensures reproducibility.
    
    Note:
        - This affects numpy, random, torch, and optionally cudnn
        - For complete reproducibility in PyTorch:
            1. Call this function at the start of your script
            2. Set num_workers=0 in DataLoaders
            3. Run on same hardware/environment
        - Some operations may still be non-deterministic on GPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Warn about operations that might still be non-deterministic
    if torch.cuda.is_available() and not deterministic_cudnn:
        print("Warning: CUDA operations may still be non-deterministic. "
              "Set deterministic_cudnn=True for full reproducibility.")


def get_device(device_str: str = None) -> torch.device:
    """Get PyTorch device based on availability and preference.
    
    Args:
        device_str: Optional device specification ('cuda', 'cpu', or None).
            If None, will use CUDA if available.
    
    Returns:
        torch.device: Device to use for computations.
    """
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device_str)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
        
    return device


def get_pandapower_net(config):
    """Get pandapower network from module path specified in config.
    
    Args:
        config: Configuration dictionary containing pandapower_module key
        
    Returns:
        pandapower.auxiliary.pandapowerNet: The network object
        
    Example:
        If config['pandapower_module'] = "pandapower.networks.case9241pegase",
        this will dynamically import and call that function.
    """
    try:
        # Support two config formats:
        # 1) pandapower_module contains full path including function: 'pandapower.networks.case1354pegase'
        # 2) pandapower_module is module and pandapower_function contains function name
        if 'pandapower_function' in config and config.get('pandapower_function'):
            module = importlib.import_module(config['pandapower_module'])
            net_func = getattr(module, config['pandapower_function'])
        else:
            # Assume full path provided in pandapower_module
            module_path, function_name = config['pandapower_module'].rsplit('.', 1)
            module = importlib.import_module(module_path)
            net_func = getattr(module, function_name)

        # Create and return the network
        return net_func()
    except ImportError:
        raise ImportError(f"Could not import pandapower module {config.get('pandapower_module')}")
    except AttributeError:
        raise AttributeError(f"Could not find network function in {config.get('pandapower_module')} or pandapower_function: {config.get('pandapower_function')}")
    except Exception as e:
        raise Exception(f"Error creating pandapower network: {str(e)}")