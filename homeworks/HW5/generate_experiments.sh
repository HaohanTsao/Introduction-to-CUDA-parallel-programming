#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw5"

# Test parameters
BLOCK_SIZES=(8 16 32)
GPU_CONFIGS=("1" "2")  # 1 GPU and 2 GPUs
LATTICE_SIZE="1024 1024"

echo "Generating Heat Diffusion experiments..."

# Loop through all parameter combinations
for gpu_count in "${GPU_CONFIGS[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        # Create experiment directory
        EXP_DIR="$WORK_DIR/exp_gpu${gpu_count}_block${block_size}x${block_size}"
        mkdir -p $EXP_DIR
        
        # Copy executable
        cp $WORK_DIR/heat_diffusion $EXP_DIR/
        
        # Create Input file
        if [ "$gpu_count" = "1" ]; then
            cat > $EXP_DIR/Input << EOL
1 1
GPU_ID
$LATTICE_SIZE
$block_size $block_size
0
EOL
        else
            cat > $EXP_DIR/Input << EOL
2 1
GPU_ID GPU_ID
$LATTICE_SIZE
$block_size $block_size
0
EOL
        fi
        
        # Create cmd file
        cat > $EXP_DIR/cmd << EOL
Universe      = vanilla
Executable    = /opt/bin/runjob
Output        = condor.out
Error         = condor.err
Log           = condor.log
Requirements  = (MTYPE == "sm61_60G")
notification  = never
Machine_count = 1

request_cpus  = $gpu_count

Initialdir = $EXP_DIR

Arguments = ./heat_diffusion Input Output

Queue
EOL
        
        echo "Generated experiment: GPU=$gpu_count, Block=${block_size}x${block_size}"
    done
done

echo "All experiment directories and files have been generated!"
echo "Total experiments: $((${#GPU_CONFIGS[@]} * ${#BLOCK_SIZES[@]}))"