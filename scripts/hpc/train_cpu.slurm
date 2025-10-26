#!/bin/bash
#SBATCH --job-name=grid_train_cpu
#SBATCH --output=train_cpu_%j.out
#SBATCH --error=train_cpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --qos=normal

# Load required modules
module load python/3.9

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/path/to/grid_security_ai

# Change to project directory
cd /path/to/grid_security_ai

# Run training
python scripts/run_training.py --config config/training.yml
