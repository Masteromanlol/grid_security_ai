#!/bin/bash
#
# setup.sh
#
# This script performs initial setup for the Grid AI project.
# It can be expanded to install local packages, set permissions,
# or download initial configuration files.
#
# This file is the successor to all `master_setup_v*.sh` scripts.

echo "Starting Grid AI Project Setup..."

# 1. Activate Conda Environment (optional, assumes already active)
# echo "Please ensure the 'grid_ai' conda environment is active."
# conda activate grid_ai

# 2. Install this project as an editable package
# This command makes `src/grid_ai` importable as `grid_ai` from anywhere
echo "Installing grid_ai package in editable mode..."
pip install -e .

# 3. Check for essential directories
echo "Creating required directories (if they don't exist)..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/sample
mkdir -p models
mkdir -p notebooks
mkdir -p logs

# 4. Set permissions on launcher scripts
echo "Setting executable permissions on launcher scripts..."
chmod +x scripts/launch/*.sh
chmod +x scripts/run_*.py

echo "--------------------------------------"
echo "Setup Complete."
echo "To run tests: pytest"
echo "To run training: python scripts/run_training.py"
echo "--------------------------------------"
