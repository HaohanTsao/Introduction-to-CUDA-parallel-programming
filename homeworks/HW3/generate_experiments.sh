#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw3"

# Lattice sizes to test
LATTICE_SIZES=(8 16 32 64)

# Loop through all lattice sizes
for L in "${LATTICE_SIZES[@]}"; do
    # Create experiment directory
    EXP_DIR="$WORK_DIR/exp_L${L}"
    mkdir -p $EXP_DIR
    
    # Copy executable
    cp $WORK_DIR/poisson3d $EXP_DIR/
    
    # Create Input file
    echo "GPU_ID" > $EXP_DIR/Input
    echo "$L" >> $EXP_DIR/Input
    
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

Arguments = ./poisson3d Input Output

Queue
EOL
    
    echo "Generated experiment for lattice size L = $L"
done

echo "All experiment directories and files have been generated!"