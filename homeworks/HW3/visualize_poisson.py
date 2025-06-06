"""
Visualization script for 3D Poisson equation results
Compares numerical solution with theoretical Coulomb potential
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def read_combined_results(filename):
    """
    Read the combined results CSV file and clean the data
    """
    try:
        # Read CSV file
        df = pd.read_csv(filename)
        
        # Clean up data - remove any non-numeric rows
        df = df[pd.to_numeric(df['L'], errors='coerce').notna()]
        df = df[pd.to_numeric(df['r'], errors='coerce').notna()]
        df = df[pd.to_numeric(df['potential'], errors='coerce').notna()]
        
        # Convert to numeric
        df['L'] = pd.to_numeric(df['L'])
        df['r'] = pd.to_numeric(df['r'])
        df['potential'] = pd.to_numeric(df['potential'])
        
        # Take absolute value of potential (handle sign issues)
        df['potential'] = df['potential'].abs()
        
        # Remove duplicates based on L and r
        df = df.drop_duplicates(subset=['L', 'r'], keep='first')
        
        # Sort by L and r
        df = df.sort_values(['L', 'r'])
        
        print(f"✅ Data processed: {len(df)} clean data points")
        print(f"📊 Available lattice sizes: {sorted(df['L'].unique())}")
        
        return df
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def coulomb_potential(r):
    """
    Calculate theoretical Coulomb potential: φ = 1/(4πr)
    """
    return 1.0 / (4.0 * np.pi * r)

def create_comparison_plots(df, save_dir="."):
    """
    Create comparison plots for numerical vs theoretical potential
    """
    # Define lattice sizes
    lattice_sizes = [8, 16, 32, 64]
    colors = ['red', 'blue', 'green', 'orange']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Plot for each lattice size
    for i, L in enumerate(lattice_sizes):
        ax = axes[i]
        
        # Filter data for this lattice size
        data_L = df[df['L'] == L]
        
        if len(data_L) > 0:
            # Sort by distance
            data_L = data_L.sort_values('r')
            
            # Extract r and potential
            r_numerical = data_L['r'].values
            phi_numerical = data_L['potential'].values
            
            # Calculate theoretical Coulomb potential
            phi_coulomb = coulomb_potential(r_numerical)
            
            # Plot numerical solution
            ax.plot(r_numerical, phi_numerical, 'o-', color=colors[i], 
                   label=f'Numerical (L={L})', linewidth=2, markersize=5)
            
            # Plot theoretical solution
            ax.plot(r_numerical, phi_coulomb, '--', color='black', 
                   label='Coulomb: 1/(4πr)', linewidth=2, alpha=0.7)
            
            # Format plot
            ax.set_xlabel('Distance r', fontsize=12)
            ax.set_ylabel('Potential φ(r)', fontsize=12)
            ax.set_title(f'Lattice Size L = {L}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xscale('log')
            
            # Add relative error text
            if len(r_numerical) > 0:
                r_test = r_numerical[0]  # First point
                phi_num = phi_numerical[0]
                phi_theo = phi_coulomb[0]
                rel_error = abs(phi_num - phi_theo) / phi_theo * 100
                ax.text(0.05, 0.95, f'Error at r={r_test:.1f}: {rel_error:.1f}%', 
                       transform=ax.transAxes, fontsize=9, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'No data for L={L}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/poisson_vs_coulomb_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot as 'poisson_vs_coulomb_comparison.png'")
    
    return fig

def create_convergence_plot(df, save_dir="."):
    """
    Create plot showing convergence to Coulomb's law as L increases
    """
    lattice_sizes = [8, 16, 32, 64]
    colors = ['red', 'blue', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    # Test distance (around r=2)
    test_r = 2.0
    tolerance = 0.5
    
    for i, L in enumerate(lattice_sizes):
        # Filter data for this lattice size
        data_L = df[df['L'] == L]
        
        if len(data_L) > 0:
            # Find data point closest to test distance
            data_near_r = data_L[abs(data_L['r'] - test_r) <= tolerance]
            
            if len(data_near_r) > 0:
                # Take the closest point
                closest_idx = (abs(data_near_r['r'] - test_r)).idxmin()
                r_actual = data_near_r.loc[closest_idx, 'r']
                phi_numerical = data_near_r.loc[closest_idx, 'potential']
                phi_coulomb = coulomb_potential(r_actual)
                
                # Plot both values
                plt.plot(L, phi_numerical, 'o-', color=colors[i], 
                        label=f'Numerical' if i==0 else "", markersize=8, linewidth=2)
                plt.plot(L, phi_coulomb, 's--', color='black', 
                        label=f'Coulomb (theory)' if i==0 else "", markersize=6, alpha=0.7)
    
    plt.xlabel('Lattice Size L', fontsize=14)
    plt.ylabel(f'Potential φ(r≈{test_r})', fontsize=14)
    plt.title('Convergence to Coulomb\'s Law with Increasing Lattice Size', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_to_coulomb.png', dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot as 'convergence_to_coulomb.png'")

def create_error_analysis(df, save_dir="."):
    """
    Create relative error analysis plot
    """
    lattice_sizes = [8, 16, 32, 64]
    
    plt.figure(figsize=(12, 8))
    
    for L in lattice_sizes:
        data_L = df[df['L'] == L]
        
        if len(data_L) > 0:
            data_L = data_L.sort_values('r')
            r_vals = data_L['r'].values
            phi_numerical = data_L['potential'].values
            phi_coulomb = coulomb_potential(r_vals)
            
            # Calculate relative error
            rel_error = abs(phi_numerical - phi_coulomb) / phi_coulomb * 100
            
            plt.plot(r_vals, rel_error, 'o-', label=f'L={L}', linewidth=2, markersize=4)
    
    plt.xlabel('Distance r', fontsize=14)
    plt.ylabel('Relative Error (%)', fontsize=14)
    plt.title('Relative Error: |φ_numerical - φ_Coulomb| / φ_Coulomb × 100%', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/relative_error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved error analysis as 'relative_error_analysis.png'")

def create_summary_table(df):
    """
    Create summary table of results
    """
    print("\n" + "="*80)
    print("SUMMARY: Numerical vs Theoretical Coulomb Potential")
    print("="*80)
    
    lattice_sizes = [8, 16, 32, 64]
    
    for L in lattice_sizes:
        data_L = df[df['L'] == L]
        
        if len(data_L) > 0:
            print(f"\nLattice Size L = {L}:")
            print("-" * 40)
            print(f"{'r':<8} {'φ_num':<12} {'φ_Coulomb':<12} {'Error(%)':<10}")
            print("-" * 40)
            
            # Sort by distance
            data_L = data_L.sort_values('r').head(6)  # Show first 6 points
            
            for _, row in data_L.iterrows():
                r = row['r']
                phi_num = row['potential']
                phi_coulomb = coulomb_potential(r)
                error = abs(phi_num - phi_coulomb) / phi_coulomb * 100
                
                print(f"{r:<8.2f} {phi_num:<12.6f} {phi_coulomb:<12.6f} {error:<10.1f}")

def main():
    """
    Main function to process and visualize Poisson 3D results
    """
    # Try to find data files
    possible_files = ["cleaned_results.csv"]
    data_file = None
    
    for filename in possible_files:
        if os.path.exists(filename):
            data_file = filename
            break
    
    if data_file is None:
        print(f"❌ Error: No data file found!")
        print(f"   Looking for: {possible_files}")
        print(f"   Please ensure one of these files is in the current directory.")
        return
    
    print("🔍 Processing 3D Poisson Equation Results...")
    print(f"📁 Reading data from: {data_file}")
    
    # Read and clean data
    df = read_combined_results(data_file)
    
    if df is None or len(df) == 0:
        print("❌ Failed to read data file or no valid data found!")
        return
    
    # Check if we have all expected lattice sizes
    expected_sizes = [8, 16, 32, 64]
    available_sizes = sorted(df['L'].unique())
    missing_sizes = [L for L in expected_sizes if L not in available_sizes]
    
    if missing_sizes:
        print(f"⚠️  Warning: Missing data for lattice sizes: {missing_sizes}")
    
    print(f"✅ Data loaded successfully: {len(df)} data points")
    print(f"📊 Lattice sizes available: {available_sizes}")
    
    # Create visualizations
    print("\n🎨 Creating comparison plots...")
    fig1 = create_comparison_plots(df)
    
    print("\n📈 Creating convergence analysis...")
    create_convergence_plot(df)
    
    print("\n📉 Creating error analysis...")
    create_error_analysis(df)
    
    # Create summary table
    create_summary_table(df)
    
    print("\n✅ All visualizations completed successfully!")
    print("\n📋 Generated files:")
    print("   • poisson_vs_coulomb_comparison.png - Main comparison plots")
    print("   • convergence_to_coulomb.png - Convergence analysis")  
    print("   • relative_error_analysis.png - Error analysis")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()