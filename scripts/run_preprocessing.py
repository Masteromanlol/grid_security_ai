#!/usr/bin/env python
"""Script for preprocessing raw simulation data."""

import argparse
from grid_ai.preprocessing import preprocess_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    preprocess_data(args.config)

if __name__ == "__main__":
    main()