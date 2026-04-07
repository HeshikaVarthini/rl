"""
Check what happened during your 400,000 timestep training run.
This will tell you if episodes completed or not.
"""
import os
import glob

def check_results():
    print("=" * 60)
    print("TRAINING RESULTS CHECKER")
    print("=" * 60)
    
    checkpoint_dir = 'checkpoints/ppo_minitaur_obstacle'
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    
    if not os.path.exists(checkpoint_dir):
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        print("Training hasn't run yet or used a different directory.")
        return
    
    print(f"\n1. Checking checkpoint directory: {checkpoint_dir}")
    print("-" * 60)
    
    # Check model files
    model_files = glob.glob(os.path.join(checkpoint_dir, '*.zip'))
    print(f"Model checkpoints found: {len(model_files)}")
    if model_files:
        print("✓ Training DID run!")
        for f in sorted(model_files)[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(model_files) > 5:
            print(f"  ... and {len(model_files) - 5} more")
    else:
        print("✗ No model checkpoints - training may not have completed")
    
    # Check plots directory
    print(f"\n2. Checking plots directory: {plots_dir}")
    print("-" * 60)
    
    if not os.path.exists(plots_dir):
        print("✗ Plots directory doesn't exist")
        print("Callback was never initialized!")
        return
    
    print("✓ Plots directory exists")
    
    # Check for plot files
    png_files = glob.glob(os.path.join(plots_dir, '*.png'))
    print(f"\nPNG plot files: {len(png_files)}")
    if png_files:
        print("✓ Plots were generated!")
        for f in sorted(png_files):
            size = os.path.getsize(f) / 1024  # KB
            print(f"  - {os.path.basename(f)} ({size:.1f} KB)")
    else:
        print("✗ No PNG files - plots were NOT generated")
    
    # Check for CSV file
    csv_file = os.path.join(plots_dir, 'training_data.csv')
    npz_file = os.path.join(plots_dir, 'training_data.npz')
    
    print(f"\n3. Checking training data...")
    print("-" * 60)
    
    if os.path.exists(csv_file):
        print(f"✓ CSV file found: training_data.csv")
        
        # Read and analyze CSV
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            print(f"  Lines in file: {len(lines)}")
            
            if len(lines) <= 1:
                print("  ✗ CSV is empty (only header or no data)")
                print("\n" + "=" * 60)
                print("DIAGNOSIS: NO EPISODES COMPLETED!")
                print("=" * 60)
                print("\nThis means:")
                print("1. Your training ran for 400,000 timesteps")
                print("2. But NO single episode finished (reached done=True)")
                print("3. Episodes are too long or never terminate")
                print("\nPossible causes:")
                print("- Episodes have very high step limits")
                print("- Robot never collides or fails")
                print("- done condition never triggers")
                print("\nSolution:")
                print("- Check your environment's termination conditions")
                print("- Add a maximum episode length limit")
                print("- Ensure collisions trigger done=True")
            else:
                print(f"  ✓ CSV has {len(lines) - 1} episodes!")
                print(f"\n  First few lines:")
                for line in lines[:5]:
                    print(f"    {line.strip()}")
                
                # Parse rewards
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    print(f"\n  Statistics:")
                    print(f"    Total episodes: {len(df)}")
                    print(f"    Average reward: {df['reward'].mean():.2f}")
                    print(f"    Best reward: {df['reward'].max():.2f}")
                    print(f"    Worst reward: {df['reward'].min():.2f}")
                    print(f"    Average length: {df['length'].mean():.1f} steps")
                    print(f"    Total timesteps: {df['timestep'].max()}")
                except:
                    pass
                
        except Exception as e:
            print(f"  ✗ Error reading CSV: {e}")
    
    elif os.path.exists(npz_file):
        print(f"✓ NumPy file found: training_data.npz")
        try:
            import numpy as np
            data = np.load(npz_file)
            episodes = data['episodes']
            rewards = data['rewards']
            print(f"  Episodes: {len(episodes)}")
            print(f"  Average reward: {np.mean(rewards):.2f}")
        except Exception as e:
            print(f"  Error reading NPZ: {e}")
    else:
        print("✗ No training data file found")
        print("\n" + "=" * 60)
        print("DIAGNOSIS: CALLBACK NOT WORKING OR NO EPISODES")
        print("=" * 60)
    
    # Check TensorBoard logs
    print(f"\n4. Checking TensorBoard logs...")
    print("-" * 60)
    
    tb_dir = os.path.join(checkpoint_dir, 'tb')
    if os.path.exists(tb_dir):
        tb_files = []
        for root, dirs, files in os.walk(tb_dir):
            tb_files.extend(files)
        print(f"✓ TensorBoard files: {len(tb_files)}")
        print(f"  View with: tensorboard --logdir={tb_dir}")
    else:
        print("✗ No TensorBoard logs")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if model_files and not png_files:
        print("\n⚠ ISSUE FOUND:")
        print("- Training ran (model checkpoints exist)")
        print("- But NO plots generated")
        print("\nMost likely cause:")
        print("→ Episodes are too long and never completed in 400,000 steps")
        print("\nNext steps:")
        print("1. Check environment max_episode_steps")
        print("2. Ensure done=True triggers on collisions")
        print("3. Consider reducing episode length")
    elif png_files:
        print("\n✓ Everything looks good!")
        print("Plots were generated successfully.")
    else:
        print("\n✗ Training may not have run properly")


if __name__ == '__main__':
    check_results()

