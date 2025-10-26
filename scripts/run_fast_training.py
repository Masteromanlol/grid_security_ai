#!/usr/bin/env python
"""Script for fast training the model (optimized for laptops)."""

import argparse
from grid_ai.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_case_9241_fast.yml",
                       help="Path to config file (defaults to fast training config)")
    args = parser.parse_args()

    train_model(args.config)

if __name__ == "__main__":
    main()
