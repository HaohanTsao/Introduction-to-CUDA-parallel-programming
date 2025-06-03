#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw5"
RESULTS_FILE="$WORK_DIR/performance_results.csv"

echo "Collecting Heat Diffusion performance results..."

# Create results file header
echo "GPU_Count,Block_Size,Processing_Time(ms),GPU_Gflops,Iterations,Total_Time(ms)" > $RESULTS_FILE

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_gpu*_block*; do
    if [ -d "$exp_dir" ]; then
        # Extract parameters from directory name
        dir_name=$(basename $exp_dir)
        gpu_count=$(echo $dir_name | sed 's/exp_gpu\([0-9]*\)_block.*/\1/')
        block_size=$(echo $dir_name | sed 's/exp_gpu[0-9]*_block\([0-9]*\)x[0-9]*/\1/')
        
        # Check if Output file exists
        if [ -f "$exp_dir/Output" ]; then
            # Extract performance data
            gpu_time=$(grep "Processing time for GPU:" $exp_dir/Output | awk '{print $5}')
            gpu_gflops=$(grep "GPU Gflops:" $exp_dir/Output | awk '{print $3}')
            iterations=$(grep "total iterations (GPU)" $exp_dir/Output | awk '{print $5}')
            total_time=$(grep "Total time for GPU:" $exp_dir/Output | awk '{print $5}')
            
            # Add to results file
            echo "$gpu_count,${block_size}x${block_size},$gpu_time,$gpu_gflops,$iterations,$total_time" >> $RESULTS_FILE
            echo "Collected: GPU=$gpu_count, Block=${block_size}x${block_size}, Time=${gpu_time}ms, Gflops=$gpu_gflops"
        else
            echo "Warning: No Output file found in $exp_dir"
        fi
    fi
done

echo "Results collected in $RESULTS_FILE"

# Create analysis summary
echo ""
echo "Performance Summary:"
echo "==================="
column -t -s, $RESULTS_FILE

# Find best performance
echo ""
echo "Best Performance Analysis:"
echo "========================="
echo "Highest Gflops:"
tail -n +2 $RESULTS_FILE | sort -t, -k4 -nr | head -1 | while IFS=, read gpu block time gflops iter total; do
    echo "  Configuration: $gpu GPU(s), $block block size -> $gflops Gflops"
done

echo "Fastest execution:"
tail -n +2 $RESULTS_FILE | sort -t, -k3 -n | head -1 | while IFS=, read gpu block time gflops iter total; do
    echo "  Configuration: $gpu GPU(s), $block block size -> ${time}ms"
done