#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw4"

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_t*_b*; do
    if [ -d "$exp_dir" ]; then
        # Extract parameters from directory name
        dir_name=$(basename $exp_dir)
        
        # Submit job
        cd $exp_dir
        condor_submit cmd
        cd $WORK_DIR
        
        echo "Submitted experiment: $dir_name"
        
        # Wait a bit to avoid submitting too many jobs at once
        sleep 2
    fi
done

echo "All jobs submitted! Use 'jview' command to monitor progress."