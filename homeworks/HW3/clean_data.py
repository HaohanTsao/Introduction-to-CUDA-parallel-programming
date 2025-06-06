"""
Clean the combined_results.csv file for Poisson 3D analysis
"""

import pandas as pd
import numpy as np

def clean_poisson_data(input_file, output_file):
    """
    Clean the raw combined results data
    """
    print(f"🔍 Reading raw data from {input_file}...")
    
    # Read the raw CSV file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Clean data
    cleaned_lines = []
    cleaned_lines.append("L,r,potential\n")  # Header
    
    seen_combinations = set()  # To avoid duplicates
    
    for line in lines[1:]:  # Skip original header
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) != 3:
            continue
            
        try:
            # Try to convert to numbers
            L = int(parts[0])
            r = float(parts[1])
            potential = float(parts[2])
            
            # Take absolute value of potential (fix sign issue)
            potential = abs(potential)
            
            # Create unique identifier for this (L, r) combination
            combination = (L, r)
            
            # Only keep if we haven't seen this combination before
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                cleaned_lines.append(f"{L},{r:.2f},{potential:.6f}\n")
                
        except ValueError:
            # Skip non-numeric lines
            continue
    
    # Write cleaned data
    with open(output_file, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"✅ Cleaned data saved to {output_file}")
    
    # Show summary
    df = pd.read_csv(output_file)
    print(f"📊 Summary:")
    print(f"   Total data points: {len(df)}")
    print(f"   Lattice sizes: {sorted(df['L'].unique())}")
    for L in sorted(df['L'].unique()):
        count = len(df[df['L'] == L])
        print(f"   L={L}: {count} points")

def main():
    input_file = "combined_results.csv"
    output_file = "cleaned_results.csv"
    
    clean_poisson_data(input_file, output_file)
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Use 'cleaned_results.csv' for visualization")
    print(f"   2. Run the visualization script with the cleaned data")

if __name__ == "__main__":
    main()