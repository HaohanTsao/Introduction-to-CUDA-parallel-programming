#!/bin/bash

# Working directory  
WORK_DIR="/home/cuda2025/B09607009/hw5"

echo "Submitting all Heat Diffusion experiments..."

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_gpu*_block*; do
    if [ -d "$exp_dir" ]; then
        # Extract parameters from directory name
        dir_name=$(basename $exp_dir)
        
        # Submit job
        cd $exp_dir
        job_output=$(condor_submit cmd)
        job_id=$(echo "$job_output" | grep "submitted to cluster" | awk '{print $6}' | cut -d. -f1)
        cd $WORK_DIR
        
        echo "Submitted $dir_name (Job ID: $job_id)"
        
        # Wait a bit to avoid submitting too many jobs at once
        sleep 1
    fi
done

echo "All jobs submitted! Use 'jview' command to monitor progress."
