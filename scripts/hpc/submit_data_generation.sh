#!/bin/bash

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.yml>"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Count number of contingencies (assuming CSV format)
N_CONT=$(tail -n +2 $(grep "contingency_file:" "$CONFIG_FILE" | cut -d' ' -f2) | wc -l)

if [ $N_CONT -eq 0 ]; then
    echo "Error: No contingencies found in contingency file!"
    exit 1
fi

# Calculate number of tasks (100 contingencies per task)
N_TASKS=$(( ($N_CONT + 99) / 100 ))

echo "Submitting job array with $N_TASKS tasks for $N_CONT contingencies..."

# Submit job array
export CONFIG_FILE
sbatch --array=1-$N_TASKS _template_data_job.slurm