#!/bin/bash
#SBATCH --job-name=grid_data_gen
#SBATCH --output=data_gen_%j.out
#SBATCH --error=data_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.9
source activate grid-ai  # Activate conda environment

python ../run_simulation_worker.py --config $CONFIG_FILE --task-id $SLURM_ARRAY_TASK_ID