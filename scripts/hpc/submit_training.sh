#!/bin/bash

# Submit training jobs for different grid cases
for config in ../../config/case_*.yml; do
    export CONFIG_FILE=$config
    sbatch _template_train_job.slurm
done