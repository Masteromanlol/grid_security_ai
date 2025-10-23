#!/usr/bin/env python
"""Script for running model diagnostics."""

import argparse
from grid_ai.evaluate import run_diagnostics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--model", required=True, help="Path to model file")
    args = parser.parse_args()
    
    run_diagnostics(args.config, args.model)

if __name__ == "__main__":
    main()