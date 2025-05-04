#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/cuda2025/hw2_matrix_trace"
# Matrix size
MATRIX_SIZE=6400

# Thread combinations
THREADS_PER_BLOCK=(32 64 128 256 512 1024)
# Block combinations
BLOCKS_PER_GRID=(128 256 512 1024)

# Loop through all parameter combinations
for threads in "${THREADS_PER_BLOCK[@]}"; do
    for blocks in "${BLOCKS_PER_GRID[@]}"; do
        # Create experiment directory
        EXP_DIR="$WORK_DIR/exp_t${threads}_b${blocks}"
        mkdir -p $EXP_DIR
        
        # Copy executable
        cp $WORK_DIR/matrixTrace $EXP_DIR/
        
        # Create Input file
        echo "GPU_ID" > $EXP_DIR/Input
        echo "$MATRIX_SIZE" >> $EXP_DIR/Input
        echo "$threads" >> $EXP_DIR/Input
        echo "$blocks" >> $EXP_DIR/Input
        
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

request_cpus  = 1

Initialdir = $EXP_DIR

Arguments = ./matrixTrace Input Output

Queue
EOL
        
        echo "Generated experiment with params: threads=$threads, blocks=$blocks"
    done
done

echo "All experiment directories and files have been generated!"