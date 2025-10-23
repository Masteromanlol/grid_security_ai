#!/usr/bin/env python
"""Test script for running a few contingency simulations."""

import os
import yaml
import logging
from grid_ai.simulation import run_simulation

def main():
    print("Starting test simulation...")
    
    # Create test config with just 5 contingencies
    print("Reading contingencies file...")
    with open('data/contingencies/contingencies_1354.txt', 'r') as f:
        header = f.readline()
        test_conts = [header] + [f.readline() for _ in range(5)]
        print(f"Read contingencies:\n{''.join(test_conts)}")
        
    # Write test contingencies
    print("Creating test contingencies file...")
    os.makedirs('data/contingencies/test', exist_ok=True)
    with open('data/contingencies/test/test_contingencies.txt', 'w') as f:
        f.writelines(test_conts)
    
    # Setup logging
    os.makedirs('logs/test_simulation', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/test_simulation/simulation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Update config for test
    print("Creating test configuration...")
    config = {
        'log_dir': 'logs/test_simulation',
        'contingency_file': 'data/contingencies/test/test_contingencies.txt',
        'output_dir': 'data/raw/test_simulation',
        'pandapower_module': 'pandapower.networks.case1354pegase',
        'simulation': {
            'method': 'ac'
        }
    }
    
    # Create config file
    print("Writing configuration file...")
    os.makedirs('config/test', exist_ok=True)
    with open('config/test/test_simulation.yml', 'w') as f:
        yaml.dump(config, f)
    
    print("Running simulation...")
    try:
        run_simulation('config/test/test_simulation.yml')
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()