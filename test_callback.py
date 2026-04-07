"""
Quick test to verify the callback and plotting work correctly.
This runs a very short training session just to test the plotting system.
"""
import os
import sys

# Set environment variables for a quick test
os.environ['NUM_ENVS'] = '1'
os.environ['TOTAL_STEPS'] = '5000'  # Just 5000 steps for testing
os.environ['SAVE_DIR'] = 'test_checkpoints'

print("=" * 60)
print("Testing Callback and Plotting System")
print("=" * 60)
print("This will run a SHORT training session (5000 steps)")
print("to verify that plots are being generated correctly.")
print("=" * 60)
print()

# Import and run the training
try:
    from train_minitaur_obstacle_rl import main
    
    print("Starting test training...")
    main()
    
    print()
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    # Check if plots were created
    plots_dir = 'test_checkpoints/plots'
    if os.path.exists(plots_dir):
        import glob
        plot_files = glob.glob(os.path.join(plots_dir, '*.png'))
        csv_files = glob.glob(os.path.join(plots_dir, '*.csv'))
        
        print(f"✓ Plots directory exists: {plots_dir}")
        print(f"✓ PNG files found: {len(plot_files)}")
        print(f"✓ CSV files found: {len(csv_files)}")
        
        if plot_files:
            print("\nPlot files created:")
            for f in plot_files:
                print(f"  - {os.path.basename(f)}")
        
        if csv_files:
            print("\nData files created:")
            for f in csv_files:
                print(f"  - {os.path.basename(f)}")
                # Try to read the CSV
                try:
                    with open(f, 'r') as file:
                        lines = file.readlines()
                        print(f"    Lines in file: {len(lines)}")
                        if len(lines) > 1:
                            print(f"    First data line: {lines[1].strip()}")
                except Exception as e:
                    print(f"    Could not read file: {e}")
        
        if not plot_files and not csv_files:
            print("\n⚠ WARNING: No plots or data files were created!")
            print("This might mean:")
            print("  1. No episodes completed during the test")
            print("  2. The callback is not working correctly")
            print("  3. Episodes are too long (increase TOTAL_STEPS)")
    else:
        print(f"✗ Plots directory not found: {plots_dir}")
        print("The callback may not be working correctly.")
    
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
