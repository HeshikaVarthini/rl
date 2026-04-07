"""
Quick test to verify matplotlib plotting works in command prompt.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def test_plotting():
    """Test if matplotlib can create and save plots."""
    print("Testing matplotlib plotting...")
    
    # Create test directory
    test_dir = 'test_plots'
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate test data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create a simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(x, y1, 'b-', label='sin(x)')
    ax1.set_title('Test Plot 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x, y2, 'r-', label='cos(x)')
    ax2.set_title('Test Plot 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(test_dir, 'test_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if os.path.exists(output_path):
        print(f"✓ SUCCESS! Plot saved to: {output_path}")
        print(f"✓ File size: {os.path.getsize(output_path)} bytes")
        print("\nMatplotlib is working correctly!")
        print("You can now run your training script and plots will be saved.")
        
        # Try to open the image
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_path)
                print(f"\nOpening test plot in default image viewer...")
        except Exception as e:
            print(f"\nCould not open image automatically: {e}")
            print(f"Please open manually: {output_path}")
        
        return True
    else:
        print("✗ FAILED! Could not create plot.")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Matplotlib Plotting Test")
    print("=" * 60)
    test_plotting()
    print("=" * 60)
