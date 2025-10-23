"""Utility functions for the grid_ai package."""

import os
import yaml
import logging
import importlib
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
        # Split module path into module and function names
        module_path, function_name = config['pandapower_module'].rsplit('.', 1)
        
        # Dynamically import module
        module = importlib.import_module(module_path)
        
        # Get the network creation function
        net_func = getattr(module, function_name)
        
        # Create and return the network
        return net_func()
    except ImportError:
        raise ImportError(f"Could not import pandapower module {config['pandapower_module']}")
    except AttributeError:
        raise AttributeError(f"Could not find network function in {config['pandapower_module']}")
    except Exception as e:
        raise Exception(f"Error creating pandapower network: {str(e)}")