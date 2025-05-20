#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw4"
RESULTS_FILE="$WORK_DIR/results_summary.csv"

# Create results file header
echo "Threads,Blocks,GPU_Processing_Time(ms),GPU_Gflops,Speedup,Accuracy" > $RESULTS_FILE

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_t*_b*; do
    if [ -d "$exp_dir" ]; then
        # Extract parameters from directory name
        dir_name=$(basename $exp_dir)
        threads=$(echo $dir_name | sed 's/exp_t\([0-9]*\)_b.*/\1/')
        blocks=$(echo $dir_name | sed 's/exp_t[0-9]*_b\([0-9]*\)/\1/')
        
        # Check if Output file exists
        if [ -f "$exp_dir/Output" ]; then
            # Extract performance data - make sure to use the exact phrases that appear in the output
            gpu_time=$(grep "Total processing time for 2 GPUs:" $exp_dir/Output | awk '{print $7}')
            gpu_gflops=$(grep "GPU Gflops:" $exp_dir/Output | awk '{print $3}')
            speedup=$(grep "Speed up of GPU vs CPU =" $exp_dir/Output | awk '{print $8}')
            accuracy=$(grep "|(h_G - h_D)/h_D| =" $exp_dir/Output | sed 's/.*= //')
            
            # Add to results file
            echo "$threads,$blocks,$gpu_time,$gpu_gflops,$speedup,$accuracy" >> $RESULTS_FILE
            echo "Collected results for params: threads=$threads, blocks=$blocks"
        else
            echo "Warning: No Output file found in $exp_dir"
        fi
    fi
done

echo "Results collected in $RESULTS_FILE"