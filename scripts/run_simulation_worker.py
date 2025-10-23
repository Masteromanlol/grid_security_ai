#!/usr/bin/env python
"""Worker script for running power grid simulations."""

import argparse
from grid_ai.simulation import run_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--task-id", type=int, required=True, help="SLURM array task ID")
    args = parser.parse_args()
    
    run_simulation(args.config, args.task_id)

if __name__ == "__main__":
    main()