"""
Script to view training plots after training is complete.
Usage: python view_training_plots.py
"""
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

def view_latest_plot(checkpoint_dir='checkpoints/ppo_minitaur_obstacle'):
    """View the latest training plot."""
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"Error: Plots directory not found: {plots_dir}")
        print("Make sure training has been run first.")
        return
    
    # Find all plot files
    plot_files = glob.glob(os.path.join(plots_dir, 'training_progress_*.png'))
    
    if not plot_files:
        print(f"No plot files found in {plots_dir}")
        return
    
    # Sort by timestep number and get the latest
    plot_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_plot = plot_files[-1]
    
    print(f"Found {len(plot_files)} training plots")
    print(f"Displaying latest plot: {os.path.basename(latest_plot)}")
    
    # Open and display the image
    img = Image.open(latest_plot)
    
    # Create a matplotlib figure to display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Training Progress - {os.path.basename(latest_plot)}', fontsize=14)
    plt.tight_layout()
    
    # Save to a temporary file and open it
    output_path = os.path.join(plots_dir, 'latest_view.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Try to open the image with default viewer
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif os.name == 'posix':  # Linux/Mac
            import subprocess
            subprocess.call(['xdg-open', output_path])
        print("Opening plot in default image viewer...")
    except Exception as e:
        print(f"Could not open image automatically: {e}")
        print(f"Please open manually: {output_path}")
    
    plt.close()


def view_all_plots(checkpoint_dir='checkpoints/ppo_minitaur_obstacle'):
    """List all available training plots."""
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"Error: Plots directory not found: {plots_dir}")
        return
    
    plot_files = glob.glob(os.path.join(plots_dir, 'training_progress_*.png'))
    plot_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"\nAvailable training plots in {plots_dir}:")
    print("=" * 60)
    for i, plot_file in enumerate(plot_files, 1):
        timestep = plot_file.split('_')[-1].split('.')[0]
        print(f"{i}. training_progress_{timestep}.png (Timestep: {timestep})")
    print("=" * 60)
    
    # Check for CSV data
    csv_file = os.path.join(plots_dir, 'training_data.csv')
    npz_file = os.path.join(plots_dir, 'training_data.npz')
    
    if os.path.exists(csv_file):
        print(f"\nTraining data CSV available: {csv_file}")
    elif os.path.exists(npz_file):
        print(f"\nTraining data (numpy) available: {npz_file}")


if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("Training Plots Viewer")
    print("=" * 60)
    
    checkpoint_dir = 'checkpoints/ppo_minitaur_obstacle'
    
    # Check if custom directory provided
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    
    # List all plots
    view_all_plots(checkpoint_dir)
    
    # View the latest plot
    print("\nOpening latest plot...")
    view_latest_plot(checkpoint_dir)
