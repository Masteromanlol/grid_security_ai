#!/bin/bash
# check_job_status.sh

# Get the job ID from the first argument
JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

# Get the status of the job
squeue -j $JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
