#!/bin/bash
#SBATCH --job-name=grid_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module load python/3.9
module load cuda/11.7

python ../run_training.py --config $CONFIG_FILE