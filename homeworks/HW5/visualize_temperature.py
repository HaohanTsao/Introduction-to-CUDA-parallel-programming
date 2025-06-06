#!/usr/bin/env python3
"""
Temperature Distribution Visualization for Heat Diffusion Problem
Reads temp_GPU.dat file and creates various visualization plots
Fixed version with proper orientation and boundary condition verification
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

def read_temperature_data(filename):
    """
    Read temperature data from the output file
    Expected format: header line followed by temperature values
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find file: {filename}")
    
    temperatures = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Skip header line and process data
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    values = [float(x) for x in line.split()]
                    if values:  # Only add non-empty rows
                        temperatures.append(values)
                except ValueError:
                    continue  # Skip malformed lines
    
    temp_array = np.array(temperatures)
    
    # Fix orientation - flip vertically so top boundary shows 400K
    temp_array = np.flipud(temp_array)
    
    return temp_array

def create_temperature_plots(temp_data, title_prefix="", save_prefix=""):
    """
    Create multiple visualization plots for temperature distribution
    """
    Ny, Nx = temp_data.shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Heatmap with colorbar
    plt.subplot(2, 3, 1)
    im1 = plt.imshow(temp_data, cmap='hot', aspect='auto', 
                     extent=[0, Nx, 0, Ny], origin='lower', vmin=270, vmax=405)
    plt.colorbar(im1, label='Temperature (K)')
    plt.title(f'{title_prefix}Temperature Distribution - Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # 2. Contour plot
    plt.subplot(2, 3, 2)
    contour_levels = np.linspace(275, 400, 15)
    contour = plt.contour(X, Y, temp_data, levels=contour_levels, colors='black', linewidths=0.8)
    contourf = plt.contourf(X, Y, temp_data, levels=20, cmap='hot', alpha=0.8)
    plt.colorbar(contourf, label='Temperature (K)')
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.0f')
    plt.title(f'{title_prefix}Temperature Distribution - Contour Plot')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # 3. 3D Surface plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    # Subsample for better 3D performance
    step = max(1, Nx // 100)
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    temp_sub = temp_data[::step, ::step]
    
    surf = ax3.plot_surface(X_sub, Y_sub, temp_sub, cmap='hot', alpha=0.8, 
                           linewidth=0, antialiased=True)
    ax3.set_title(f'{title_prefix}Temperature Distribution - 3D Surface')
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.set_zlabel('Temperature (K)')
    ax3.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=20)
    
    # 4. Cross-section at middle
    plt.subplot(2, 3, 4)
    mid_y = Ny // 2
    mid_x = Nx // 2
    plt.plot(x, temp_data[mid_y, :], 'r-', linewidth=2, label=f'Horizontal (Y = {mid_y})')
    plt.plot(y, temp_data[:, mid_x], 'b-', linewidth=2, label=f'Vertical (X = {mid_x})')
    plt.xlabel('Coordinate')
    plt.ylabel('Temperature (K)')
    plt.title(f'{title_prefix}Cross-sections through Center')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(270, 405)
    
    # 5. Temperature gradient magnitude
    plt.subplot(2, 3, 5)
    grad_y, grad_x = np.gradient(temp_data)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    im5 = plt.imshow(grad_magnitude, cmap='viridis', aspect='auto',
                     extent=[0, Nx, 0, Ny], origin='lower')
    plt.colorbar(im5, label='|∇T| (K/unit)')
    plt.title(f'{title_prefix}Temperature Gradient Magnitude')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # 6. Statistics and boundary verification
    plt.subplot(2, 3, 6)
    
    # Calculate boundary averages
    top_boundary = temp_data[-1, :].mean()      # Top row
    bottom_boundary = temp_data[0, :].mean()    # Bottom row
    left_boundary = temp_data[:, 0].mean()      # Left column
    right_boundary = temp_data[:, -1].mean()    # Right column
    
    # Interior temperature statistics
    interior_temp = temp_data[1:-1, 1:-1]  # Exclude boundaries
    
    plt.text(0.05, 0.95, f'Temperature Statistics:', fontsize=14, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'Min Temperature: {temp_data.min():.1f} K', fontsize=11,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, f'Max Temperature: {temp_data.max():.1f} K', fontsize=11,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, f'Mean Temperature: {temp_data.mean():.1f} K', fontsize=11,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.70, f'Interior Mean: {interior_temp.mean():.1f} K', fontsize=11,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.65, f'Std Temperature: {temp_data.std():.1f} K', fontsize=11,
             transform=plt.gca().transAxes)
    
    plt.text(0.05, 0.55, f'Boundary Verification:', fontsize=14, weight='bold',
             transform=plt.gca().transAxes)
    
    # Color code boundary verification
    top_color = 'green' if abs(top_boundary - 400.0) < 5.0 else 'red'
    bottom_color = 'green' if abs(bottom_boundary - 273.0) < 5.0 else 'red'
    left_color = 'green' if abs(left_boundary - 273.0) < 5.0 else 'red'
    right_color = 'green' if abs(right_boundary - 273.0) < 5.0 else 'red'
    
    plt.text(0.05, 0.45, f'Top edge: {top_boundary:.1f} K (expected: 400.0)', 
             fontsize=11, color=top_color, transform=plt.gca().transAxes)
    plt.text(0.05, 0.40, f'Bottom edge: {bottom_boundary:.1f} K (expected: 273.0)', 
             fontsize=11, color=bottom_color, transform=plt.gca().transAxes)
    plt.text(0.05, 0.35, f'Left edge: {left_boundary:.1f} K (expected: 273.0)', 
             fontsize=11, color=left_color, transform=plt.gca().transAxes)
    plt.text(0.05, 0.30, f'Right edge: {right_boundary:.1f} K (expected: 273.0)', 
             fontsize=11, color=right_color, transform=plt.gca().transAxes)
    
    # Physical validation
    plt.text(0.05, 0.20, f'Physical Validation:', fontsize=14, weight='bold',
             transform=plt.gca().transAxes)
    
    # Check if temperature decreases from top to bottom
    temp_gradient_ok = top_boundary > bottom_boundary
    gradient_color = 'green' if temp_gradient_ok else 'red'
    plt.text(0.05, 0.10, f'Temperature Gradient: {"✓ Correct" if temp_gradient_ok else "✗ Wrong"}', 
             fontsize=11, color=gradient_color, transform=plt.gca().transAxes)
    
    # Check temperature range
    range_ok = 270 <= temp_data.min() <= 275 and 395 <= temp_data.max() <= 405
    range_color = 'green' if range_ok else 'red'
    plt.text(0.05, 0.05, f'Temperature Range: {"✓ Physical" if range_ok else "✗ Unphysical"}', 
             fontsize=11, color=range_color, transform=plt.gca().transAxes)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    if save_prefix:
        filename = f'{save_prefix}_temperature_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved detailed analysis as '{filename}'")
    
    return fig

def compare_initial_final(initial_file, final_file, save_prefix=""):
    """
    Compare initial and final temperature distributions
    """
    try:
        temp_initial = read_temperature_data(initial_file)
        temp_final = read_temperature_data(final_file)
        
        # Ensure both arrays have the same shape
        if temp_initial.shape != temp_final.shape:
            print(f"Warning: Shape mismatch - Initial: {temp_initial.shape}, Final: {temp_final.shape}")
            min_shape = (min(temp_initial.shape[0], temp_final.shape[0]),
                        min(temp_initial.shape[1], temp_final.shape[1]))
            temp_initial = temp_initial[:min_shape[0], :min_shape[1]]
            temp_final = temp_final[:min_shape[0], :min_shape[1]]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Initial distribution
        im1 = axes[0].imshow(temp_initial, cmap='hot', aspect='auto', origin='lower',
                            vmin=270, vmax=405)
        axes[0].set_title('Initial Temperature Distribution', fontsize=14)
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Temperature (K)')
        
        # Final distribution
        im2 = axes[1].imshow(temp_final, cmap='hot', aspect='auto', origin='lower',
                            vmin=270, vmax=405)
        axes[1].set_title('Final Temperature Distribution (Steady State)', fontsize=14)
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Temperature (K)')
        
        # Difference
        temp_diff = temp_final - temp_initial
        max_diff = max(abs(temp_diff.min()), abs(temp_diff.max()))
        im3 = axes[2].imshow(temp_diff, cmap='coolwarm', aspect='auto', origin='lower',
                            vmin=-max_diff, vmax=max_diff)
        axes[2].set_title('Temperature Change (Final - Initial)', fontsize=14)
        axes[2].set_xlabel('X coordinate')
        axes[2].set_ylabel('Y coordinate')
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label('ΔT (K)')
        
        plt.tight_layout()
        
        # Save the figure
        if save_prefix:
            filename = f'{save_prefix}_temperature_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved comparison as '{filename}'")
        
        return fig
        
    except FileNotFoundError as e:
        print(f"Warning: Could not find file for comparison: {e}")
        return None

def analyze_multiple_configurations(base_dir):
    """
    Analyze and compare multiple experimental configurations
    """
    configurations = [
        ("exp_gpu1_block16x16", "Single GPU, 16×16 blocks"),
        ("exp_gpu2_block16x16", "Dual GPU, 16×16 blocks (Best)"),
        ("exp_gpu1_block32x32", "Single GPU, 32×32 blocks"),
        ("exp_gpu2_block32x32", "Dual GPU, 32×32 blocks")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (config_dir, config_name) in enumerate(configurations):
        gpu_file = f"{base_dir}/{config_dir}/temp_GPU.dat"
        
        if os.path.exists(gpu_file):
            try:
                temp_data = read_temperature_data(gpu_file)
                
                im = axes[i].imshow(temp_data, cmap='hot', aspect='auto', origin='lower',
                                  vmin=270, vmax=405)
                axes[i].set_title(config_name, fontsize=12)
                axes[i].set_xlabel('X coordinate')
                axes[i].set_ylabel('Y coordinate')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i])
                cbar.set_label('Temperature (K)')
                
                # Add boundary verification text
                top_temp = temp_data[-1, :].mean()
                bottom_temp = temp_data[0, :].mean()
                axes[i].text(0.02, 0.98, f'Top: {top_temp:.1f}K\nBottom: {bottom_temp:.1f}K', 
                           transform=axes[i].transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
                           
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error loading\n{config_dir}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                print(f"Error processing {config_dir}: {e}")
        else:
            axes[i].text(0.5, 0.5, f'File not found:\n{config_dir}', 
                       transform=axes[i].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/all_configurations_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved multi-configuration comparison as 'all_configurations_comparison.png'")
    
    return fig

def main():
    """
    Main function to process and visualize temperature data
    """
    # Base directory for your downloaded files
    base_dir = "path_to/hw5"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found!")
        print("Please update the base_dir path in the script.")
        return
    
    # Use the best configuration (2 GPU, 16x16 block)
    best_config = "exp_gpu2_block16x16"
    gpu_result_file = f"{base_dir}/{best_config}/temp_GPU.dat"
    initial_file = f"{base_dir}/{best_config}/temp_initial.dat"
    
    print("🔍 Processing Heat Diffusion Temperature Distribution...")
    print(f"📁 Base directory: {base_dir}")
    print(f"🏆 Best configuration: {best_config}")
    
    try:
        # Read and validate data
        print("\n📊 Reading temperature data...")
        temp_data = read_temperature_data(gpu_result_file)
        print(f"✅ Data loaded successfully: {temp_data.shape} grid")
        print(f"🌡️  Temperature range: {temp_data.min():.1f} - {temp_data.max():.1f} K")
        
        # Quick validation
        top_temp = temp_data[-1, :].mean()
        bottom_temp = temp_data[0, :].mean()
        print(f"🔺 Top boundary average: {top_temp:.1f} K (expected: ~400 K)")
        print(f"🔻 Bottom boundary average: {bottom_temp:.1f} K (expected: ~273 K)")
        
        # Create comprehensive analysis plots
        print("\n🎨 Creating detailed temperature analysis...")
        fig1 = create_temperature_plots(temp_data, "GPU Solution - ", 
                                      f"{base_dir}/{best_config}")
        
        # Compare initial vs final
        print("\n📈 Creating initial vs final comparison...")
        fig2 = compare_initial_final(initial_file, gpu_result_file, 
                                   f"{base_dir}/{best_config}")
        
        # Analyze multiple configurations if requested
        print("\n🔄 Creating multi-configuration comparison...")
        fig3 = analyze_multiple_configurations(base_dir)
        
        print("\n✅ All visualizations completed successfully!")
        print("\n📋 Generated files:")
        print(f"   • {best_config}_temperature_analysis.png - Detailed analysis")
        print(f"   • {best_config}_temperature_comparison.png - Initial vs Final")
        print(f"   • all_configurations_comparison.png - Multi-config comparison")
        
        # Show the plots
        plt.show()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find temperature data file!")
        print(f"   Missing file: {e}")
        print(f"   Please ensure the files exist in: {base_dir}/{best_config}/")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()