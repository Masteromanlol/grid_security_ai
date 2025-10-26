import subprocess
import os
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st

def run_command(cmd, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def run_pipeline_script(script_name, config_path, cwd=None):
    """Run a pipeline script with given config."""
    if cwd is None:
        cwd = Path(__file__).parent.parent

    cmd = f"python scripts/{script_name} --config {config_path}"
    success, stdout, stderr = run_command(cmd, cwd)

    return success, stdout, stderr

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config {config_path}: {e}")
        return None

def get_available_configs():
    """Get list of available configuration files."""
    config_dir = Path("config")
    if not config_dir.exists():
        return []

    return [str(f) for f in config_dir.glob("*.yml")]

def get_log_files():
    """Get list of available log files."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []

    return list(logs_dir.rglob("*.log"))

def get_result_files():
    """Get list of available result files."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []

    return list(results_dir.rglob("*"))

def get_plot_files():
    """Get list of available plot files."""
    plots_dir = Path("plots")
    if not plots_dir.exists():
        return []

    return list(plots_dir.rglob("*.png"))

def check_system_status():
    """Check the status of different system components."""
    status = {}

    # Check if data directories exist
    status['data_raw'] = Path("data/raw").exists() and any(Path("data/raw").iterdir())
    status['data_processed'] = Path("data/processed").exists() and any(Path("data/processed").iterdir())

    # Check if models exist
    status['models'] = Path("models").exists() and any(Path("models").iterdir())

    # Check if results exist
    status['results'] = Path("results").exists() and any(Path("results").iterdir())

    # Check if logs exist
    status['logs'] = Path("logs").exists() and any(Path("logs").iterdir())

    return status

def get_pipeline_status():
    """Get the current status of the pipeline."""
    status = check_system_status()

    pipeline_steps = {
        'simulate': status.get('data_raw', False),
        'preprocess': status.get('data_processed', False),
        'train': status.get('models', False),
        'evaluate': status.get('results', False)
    }

    return pipeline_steps

def format_log_content(log_content, max_lines=100):
    """Format log content for display."""
    lines = log_content.split('\n')
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        lines.insert(0, f"... (showing last {max_lines} lines)")

    return '\n'.join(lines)

def get_log_stats(log_content):
    """Get statistics from log content."""
    lines = log_content.split('\n')
    stats = {
        'total_lines': len(lines),
        'errors': sum(1 for line in lines if 'ERROR' in line.upper()),
        'warnings': sum(1 for line in lines if 'WARNING' in line.upper()),
        'info': sum(1 for line in lines if 'INFO' in line.upper())
    }
    return stats
