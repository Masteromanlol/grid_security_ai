#!/usr/bin/env python
"""Unified pipeline runner for Grid Security AI.

This script runs the complete pipeline: simulate → preprocess → train → evaluate
with a single command and easy configuration.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grid_ai import utils
from grid_ai.simulation import run_simulation
from grid_ai.preprocessing import preprocess_data
from grid_ai.train import train_model
from grid_ai.evaluate import evaluate_model, run_diagnostics

logger = logging.getLogger(__name__)

def validate_pipeline_config(config: Dict[str, Any]) -> None:
    """Validate pipeline configuration."""
    required_keys = ['pipeline', 'case', 'simulation', 'preprocessing', 'training', 'evaluation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    steps = config['pipeline']['steps']
    valid_steps = ['simulate', 'preprocess', 'train', 'evaluate']
    for step in steps:
        if step not in valid_steps:
            raise ValueError(f"Invalid pipeline step: {step}. Valid steps: {valid_steps}")

def create_step_configs(pipeline_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create individual step configurations from pipeline config."""

    case = pipeline_config['pipeline']['case']
    mode = pipeline_config['pipeline'].get('mode', 'fast')

    # Simulation config
    sim_config = {
        'log_dir': f"logs/simulation_{case}",
        'contingency_file': pipeline_config['case']['contingency_file'],
        'output_dir': pipeline_config['simulation']['output_dir'],
        'pandapower_module': pipeline_config['case']['grid_module'],
        'simulation': pipeline_config['simulation']
    }

    # Preprocessing config
    preprocess_config = {
        'log_dir': f"logs/preprocessing_{case}",
        'processed_dataset_file': pipeline_config['preprocessing']['processed_dataset_file'],
        'normalization_params_file': pipeline_config['preprocessing']['normalization_params_file'],
        'raw_simulation_dirs': pipeline_config['preprocessing']['raw_simulation_dirs'],
        'pandapower_module': pipeline_config['case']['grid_module'],
        'chunk_size': pipeline_config['preprocessing']['chunk_size']
    }

    # Training config
    train_config = pipeline_config['training'].copy()
    if mode == 'fast':
        train_config.update({
            'log_dir': f"logs/training_{case}_fast",
            'model_save_dir': f"models/checkpoints_{case}_fast"
        })

    # Evaluation config
    eval_config = pipeline_config['evaluation'].copy()

    return {
        'simulate': sim_config,
        'preprocess': preprocess_config,
        'train': train_config,
        'evaluate': eval_config
    }

def run_step(step_name: str, step_config: Dict[str, Any], task_id: int = None) -> bool:
    """Run a single pipeline step."""
    logger.info(f"Starting step: {step_name}")

    try:
        if step_name == 'simulate':
            # Create temporary config file for simulation
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(step_config, f)
                temp_config = f.name
            try:
                run_simulation(temp_config, task_id=task_id)
            finally:
                os.unlink(temp_config)
        elif step_name == 'preprocess':
            # Create temporary config file for preprocessing
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(step_config, f)
                temp_config = f.name
            try:
                preprocess_data(temp_config)
            finally:
                os.unlink(temp_config)
        elif step_name == 'train':
            # Create temporary config file for training
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(step_config, f)
                temp_config = f.name
            try:
                train_model(temp_config)
            finally:
                os.unlink(temp_config)
        elif step_name == 'evaluate':
            model_path = step_config.get('model_path')
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}, skipping evaluation")
                return False
            # Create temporary config file for evaluation
            import tempfile
            import yaml
            eval_config = step_config.copy()
            eval_config['test_dataset_file'] = step_config.get('test_dataset_file')
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(eval_config, f)
                temp_config = f.name
            try:
                evaluate_model(temp_config, model_path)
                run_diagnostics(temp_config, model_path)
            finally:
                os.unlink(temp_config)
        else:
            raise ValueError(f"Unknown step: {step_name}")

        logger.info(f"Completed step: {step_name}")
        return True

    except Exception as e:
        logger.error(f"Failed step {step_name}: {str(e)}")
        return False

def run_pipeline(config_path: str, steps: List[str] = None, task_id: int = None) -> bool:
    """Run the complete pipeline."""

    # Load and validate config
    config = utils.load_config(config_path)
    validate_pipeline_config(config)

    # Override steps if specified
    if steps:
        config['pipeline']['steps'] = steps

    pipeline_steps = config['pipeline']['steps']
    logger.info(f"Running pipeline steps: {pipeline_steps}")

    # Create step-specific configs
    step_configs = create_step_configs(config)

    # Run each step
    success = True
    for step in pipeline_steps:
        if not run_step(step, step_configs[step], task_id=task_id):
            success = False
            if config['pipeline'].get('fail_fast', True):
                logger.error("Pipeline failed, stopping...")
                break

    return success

def main():
    parser = argparse.ArgumentParser(description="Run Grid Security AI Pipeline")
    parser.add_argument("--config", default="config/pipeline.yml",
                       help="Path to pipeline config file")
    parser.add_argument("--steps", nargs="+",
                       choices=['simulate', 'preprocess', 'train', 'evaluate'],
                       help="Specific steps to run (overrides config)")
    parser.add_argument("--task-id", type=int,
                       help="Task ID for parallel simulation")
    parser.add_argument("--log-level", default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Setup project logging
    log_dir = "logs/pipeline"
    os.makedirs(log_dir, exist_ok=True)
    utils.setup_logging(log_dir, 'pipeline')

    logger.info("Starting Grid Security AI Pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Steps: {args.steps or 'from config'}")

    try:
        success = run_pipeline(args.config, args.steps, args.task_id)
        if success:
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
    except Exception as e:
        logger.error(f"Pipeline crashed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
