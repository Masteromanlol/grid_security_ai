#!/bin/bash
#SBATCH --job-name=grid_train_case_9241
#SBATCH --output=logs/hpc/train_case_9241_%j.out
#SBATCH --error=logs/hpc/train_case_9241_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=normal

# Load required modules
module load python/3.9
module load cuda/11.7

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/path/to/grid_security_ai

# Change to project directory
cd /path/to/grid_security_ai

# Run training
python scripts/run_training.py --config config/training_case_9241.yml
