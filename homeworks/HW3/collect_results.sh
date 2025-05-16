#!/bin/bash

# Working directory
WORK_DIR="/home/cuda2025/B09607009/hw3"
RESULTS_DIR="$WORK_DIR/results"
COMBINED_CSV="$RESULTS_DIR/combined_results.csv"

# Create results directory
mkdir -p $RESULTS_DIR

# Create combined CSV file with header
echo "L,r,potential" > $COMBINED_CSV

# Loop through all experiment directories
for exp_dir in $WORK_DIR/exp_L*; do
    if [ -d "$exp_dir" ]; then
        # Extract lattice size from directory name
        dir_name=$(basename $exp_dir)
        L=$(echo $dir_name | sed 's/exp_L\([0-9]*\)/\1/')
        
        # Check if Output file exists
        if [ -f "$exp_dir/Output" ]; then
            # Create individual results file
            RESULT_FILE="$RESULTS_DIR/potential_L${L}.csv"
            
            # Extract potential vs distance data
            grep -A 100 "r,phi(r)" $exp_dir/Output | grep -v "r,phi(r)" | grep "," > $RESULT_FILE
            
            # Add to combined results file with L prefix
            while IFS=, read -r r phi; do
                echo "$L,$r,$phi" >> $COMBINED_CSV
            done < $RESULT_FILE
            
            echo "Collected results for lattice size L = $L"
            
            # Also copy the full output file for reference
            cp $exp_dir/Output $RESULTS_DIR/full_output_L${L}.txt
        else
            echo "Warning: No Output file found in $exp_dir"
        fi
    fi
done

echo "Results collected in $RESULTS_DIR directory"
echo "Combined results available in $COMBINED_CSV"