#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/cuda2025/hw2_matrix_trace"
RESULTS_FILE="$WORK_DIR/results_summary.csv"

# Create results file header
echo "Threads,Blocks,GPU_Processing_Time(ms),GPU_Gflops,Total_GPU_Time(ms),Speedup" > $RESULTS_FILE

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_t*_b*; do
    if [ -d "$exp_dir" ]; then
        # Extract parameters from directory name
        dir_name=$(basename $exp_dir)
        threads=$(echo $dir_name | sed 's/exp_t\([0-9]*\)_b.*/\1/')
        blocks=$(echo $dir_name | sed 's/exp_t[0-9]*_b\([0-9]*\)/\1/')
        
        # Check if Output file exists
        if [ -f "$exp_dir/Output" ]; then
            # Extract performance data
            gpu_time=$(grep "Processing time for GPU" $exp_dir/Output | awk '{print $5}')
            gpu_gflops=$(grep "GPU Gflops" $exp_dir/Output | awk '{print $3}')
            total_time=$(grep "Total time for GPU" $exp_dir/Output | awk '{print $5}')
            speedup=$(grep "Speed up of GPU" $exp_dir/Output | awk '{print $6}')
            
            # Add to results file
            echo "$threads,$blocks,$gpu_time,$gpu_gflops,$total_time,$speedup" >> $RESULTS_FILE
            echo "Collected results for params: threads=$threads, blocks=$blocks"
        else
            echo "Warning: No Output file found in $exp_dir"
        fi
    fi
done

echo "Results collected in $RESULTS_FILE"