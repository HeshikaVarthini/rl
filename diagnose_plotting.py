"""
Diagnostic script to check why plots aren't being generated.
Run this to identify the issue.
"""
import os
import sys
import glob

def check_directory_structure():
    """Check if directories exist and have correct permissions."""
    print("\n1. Checking Directory Structure...")
    print("-" * 60)
    
    checkpoint_dir = 'checkpoints/ppo_minitaur_obstacle'
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    
    if os.path.exists(checkpoint_dir):
        print(f"✓ Checkpoint directory exists: {checkpoint_dir}")
    else:
        print(f"✗ Checkpoint directory NOT found: {checkpoint_dir}")
        return False
    
    if os.path.exists(plots_dir):
        print(f"✓ Plots directory exists: {plots_dir}")
    else:
        print(f"✗ Plots directory NOT found: {plots_dir}")
        print("  → The callback may not have been initialized")
        return False
    
    # Check if we can write to the directory
    try:
        test_file = os.path.join(plots_dir, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"✓ Can write to plots directory")
    except Exception as e:
        print(f"✗ Cannot write to plots directory: {e}")
        return False
    
    return True


def check_files():
    """Check what files exist in the plots directory."""
    print("\n2. Checking Files...")
    print("-" * 60)
    
    plots_dir = 'checkpoints/ppo_minitaur_obstacle/plots'
    
    if not os.path.exists(plots_dir):
        print(f"✗ Plots directory doesn't exist: {plots_dir}")
        return False
    
    # Check for PNG files
    png_files = glob.glob(os.path.join(plots_dir, '*.png'))
    print(f"PNG files found: {len(png_files)}")
    if png_files:
        for f in png_files:
            size = os.path.getsize(f)
            print(f"  - {os.path.basename(f)} ({size} bytes)")
    
    # Check for CSV files
    csv_files = glob.glob(os.path.join(plots_dir, '*.csv'))
    print(f"CSV files found: {len(csv_files)}")
    if csv_files:
        for f in csv_files:
            size = os.path.getsize(f)
            print(f"  - {os.path.basename(f)} ({size} bytes)")
            
            # Try to read CSV
            try:
                with open(f, 'r') as file:
                    lines = file.readlines()
                    print(f"    Lines: {len(lines)}")
                    if len(lines) > 0:
                        print(f"    Header: {lines[0].strip()}")
                    if len(lines) > 1:
                        print(f"    First data: {lines[1].strip()}")
            except Exception as e:
                print(f"    Error reading: {e}")
    
    # Check for NPZ files
    npz_files = glob.glob(os.path.join(plots_dir, '*.npz'))
    print(f"NPZ files found: {len(npz_files)}")
    if npz_files:
        for f in npz_files:
            size = os.path.getsize(f)
            print(f"  - {os.path.basename(f)} ({size} bytes)")
    
    if not png_files and not csv_files and not npz_files:
        print("\n⚠ No files found in plots directory!")
        return False
    
    return True


def check_training_logs():
    """Check if training has actually run."""
    print("\n3. Checking Training Evidence...")
    print("-" * 60)
    
    checkpoint_dir = 'checkpoints/ppo_minitaur_obstacle'
    
    # Check for model checkpoints
    model_files = glob.glob(os.path.join(checkpoint_dir, '*.zip'))
    print(f"Model checkpoint files: {len(model_files)}")
    if model_files:
        for f in model_files[:5]:  # Show first 5
            print(f"  - {os.path.basename(f)}")
        if len(model_files) > 5:
            print(f"  ... and {len(model_files) - 5} more")
    else:
        print("  ✗ No model checkpoints found - training may not have run")
    
    # Check for tensorboard logs
    tb_dir = os.path.join(checkpoint_dir, 'tb')
    if os.path.exists(tb_dir):
        tb_files = []
        for root, dirs, files in os.walk(tb_dir):
            tb_files.extend([os.path.join(root, f) for f in files])
        print(f"TensorBoard log files: {len(tb_files)}")
        if tb_files:
            print("  ✓ TensorBoard logs exist")
    else:
        print("  ✗ No TensorBoard logs found")
    
    return len(model_files) > 0


def check_matplotlib():
    """Check if matplotlib is working."""
    print("\n4. Checking Matplotlib...")
    print("-" * 60)
    
    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        print(f"✓ Backend: {matplotlib.get_backend()}")
        
        # Try to create a simple plot
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        test_file = 'test_matplotlib.png'
        plt.savefig(test_file)
        plt.close()
        
        if os.path.exists(test_file):
            print(f"✓ Can create plots (test file: {test_file})")
            os.remove(test_file)
            return True
        else:
            print("✗ Cannot create plots")
            return False
            
    except Exception as e:
        print(f"✗ Matplotlib error: {e}")
        return False


def provide_recommendations():
    """Provide recommendations based on findings."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    plots_dir = 'checkpoints/ppo_minitaur_obstacle/plots'
    
    if not os.path.exists(plots_dir):
        print("\n1. Plots directory doesn't exist:")
        print("   → Run training first: python train_minitaur_obstacle_rl.py")
        print("   → The directory will be created automatically")
        return
    
    png_files = glob.glob(os.path.join(plots_dir, '*.png'))
    csv_files = glob.glob(os.path.join(plots_dir, '*.csv'))
    
    if not png_files and not csv_files:
        print("\n1. No plots or data files found:")
        print("   Possible causes:")
        print("   a) Training hasn't run long enough for episodes to complete")
        print("      → Episodes need to finish before plots are generated")
        print("      → Try running for at least 10,000 timesteps")
        print()
        print("   b) Episodes are very long")
        print("      → Check if your environment episodes are completing")
        print("      → Look for 'Episode X: Reward=...' messages in console")
        print()
        print("   c) Callback not working")
        print("      → Run: python test_callback.py")
        print("      → This will test the callback system")
        print()
        print("2. Quick test:")
        print("   python test_callback.py")
        return
    
    if csv_files:
        # Check if CSV has data
        with open(csv_files[0], 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= 1:
            print("\n1. CSV file exists but has no data:")
            print("   → No episodes have completed yet")
            print("   → Continue training and wait for episodes to finish")
        else:
            print("\n✓ Everything looks good!")
            print(f"  - {len(png_files)} plot files")
            print(f"  - {len(lines) - 1} episodes recorded")
            print("\nTo view plots:")
            print("  python view_training_plots.py")


def main():
    print("=" * 60)
    print("PLOTTING DIAGNOSTIC TOOL")
    print("=" * 60)
    print("This tool will check why plots aren't being generated.")
    
    results = []
    
    results.append(check_directory_structure())
    results.append(check_files())
    results.append(check_training_logs())
    results.append(check_matplotlib())
    
    provide_recommendations()
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All checks passed!")
    else:
        print("⚠ Some issues found - see recommendations above")
    print("=" * 60)


if __name__ == '__main__':
    main()
