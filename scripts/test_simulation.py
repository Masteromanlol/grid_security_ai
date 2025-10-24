#!/usr/bin/env python
"""Test script for running a few contingency simulations."""

from grid_ai.simulation import run_simulation
import os
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        # Use provided config
        config_path = sys.argv[2]
        run_simulation(config_path)
    else:
        # Create test config with just 5 contingencies
        with open('data/contingencies/contingencies_1354.txt', 'r') as f:
            header = f.readline()
            test_conts = [header] + [f.readline() for _ in range(5)]
            
        # Write test contingencies
        os.makedirs('data/contingencies/test', exist_ok=True)
        with open('data/contingencies/test/test_contingencies.txt', 'w') as f:
            f.writelines(test_conts)
        
        # Update config for test
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
        os.makedirs('config/test', exist_ok=True)
        import yaml
        with open('config/test/test_simulation.yml', 'w') as f:
            yaml.dump(config, f)
        
        # Run simulation
        run_simulation('config/test/test_simulation.yml')
