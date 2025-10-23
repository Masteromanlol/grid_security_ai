#!/usr/bin/env python
"""Script for training the model."""

import argparse
from grid_ai.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    train_model(args.config)

if __name__ == "__main__":
    main()