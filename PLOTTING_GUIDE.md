# Training Plots Guide

## Overview
The training script now automatically generates and saves comprehensive training plots to help you monitor your robot's learning progress.

## Features
- ✅ **Episode Rewards** - Track total rewards per episode
- ✅ **Episode Lengths** - Monitor how long episodes last
- ✅ **Reward Distribution** - See reward consistency
- ✅ **Training Statistics** - View mean, std, max, min rewards
- ✅ **Moving Averages** - Smooth trends for better visualization
- ✅ **CSV Export** - Raw data for custom analysis

## Quick Start

### 1. Test Plotting (Optional)
First, verify that matplotlib works on your system:
```bash
python test_plotting.py
```
This will create a test plot in `test_plots/` directory.

### 2. Run Training
Start your training as usual:
```bash
python train_minitaur_obstacle_rl.py
```

### 3. Monitor Progress
During training, plots are automatically saved every 5,000 timesteps to:
```
checkpoints/ppo_minitaur_obstacle/plots/
```

### 4. View Plots
After training (or during training), view the latest plot:
```bash
python view_training_plots.py
```

## File Structure
After training, you'll find:
```
checkpoints/ppo_minitaur_obstacle/
├── plots/
│   ├── training_progress_5000.png
│   ├── training_progress_10000.png
│   ├── training_progress_15000.png
│   ├── ...
│   ├── training_data.csv          # Raw data
│   └── latest_view.png            # Latest plot view
├── ppo_minitaur_obs_10000.zip     # Model checkpoints
├── ppo_minitaur_obs_20000.zip
├── ppo_minitaur_obs_final.zip     # Final model
└── tb/                             # TensorBoard logs
    └── PPO_1/
```

## Understanding the Plots

### Plot 1: Episode Rewards (Top Left)
- **Blue line**: Raw episode rewards
- **Red line**: Moving average (smoothed trend)
- **What to look for**: Upward trend = learning is working!

### Plot 2: Episode Lengths (Top Right)
- **Green line**: Steps per episode
- **Red line**: Moving average
- **What to look for**: Longer episodes = robot survives longer

### Plot 3: Reward Distribution (Bottom Left)
- **Histogram**: Distribution of all episode rewards
- **Red dashed line**: Mean reward
- **Green dashed line**: Median reward
- **What to look for**: Distribution shifting right = improvement

### Plot 4: Training Statistics (Bottom Right)
- Total episodes completed
- Total timesteps trained
- Mean, Std, Max, Min rewards
- Average episode length

## Customization

### Change Plot Frequency
Edit `train_minitaur_obstacle_rl.py`:
```python
reward_loss_callback = RewardLossCallback(
    save_dir=save_dir,
    plot_freq=5_000,  # Change this value (in timesteps)
    verbose=1
)
```

### Change Save Directory
Set environment variable before running:
```bash
# Windows Command Prompt
set SAVE_DIR=my_custom_dir
python train_minitaur_obstacle_rl.py

# Windows PowerShell
$env:SAVE_DIR="my_custom_dir"
python train_minitaur_obstacle_rl.py
```

## Troubleshooting

### Issue: No plots are being created
**Solution**: 
1. Run `python test_plotting.py` to verify matplotlib works
2. Check that the `plots/` directory is being created
3. Ensure you have write permissions in the checkpoint directory

### Issue: "No module named 'PIL'"
**Solution**: Install Pillow for viewing plots
```bash
pip install Pillow
```

### Issue: "No module named 'pandas'"
**Solution**: Install pandas for CSV export (optional)
```bash
pip install pandas
```
If pandas is not available, data will be saved as `.npz` (numpy format) instead.

### Issue: Plots are blank or corrupted
**Solution**: 
1. Make sure training has run for at least a few episodes
2. Check console output for any error messages
3. Verify the plot files are not 0 bytes in size

## Viewing Plots in Command Prompt

Since you're using command prompt, plots are saved to disk automatically. To view them:

**Option 1**: Use the viewer script
```bash
python view_training_plots.py
```

**Option 2**: Open manually
Navigate to the plots folder and double-click any PNG file:
```
checkpoints\ppo_minitaur_obstacle\plots\training_progress_XXXXX.png
```

**Option 3**: Use TensorBoard (for real-time monitoring)
```bash
tensorboard --logdir=checkpoints/ppo_minitaur_obstacle/tb
```
Then open your browser to `http://localhost:6006`

## Advanced: Custom Analysis

### Load CSV Data
```python
import pandas as pd

# Load training data
df = pd.read_csv('checkpoints/ppo_minitaur_obstacle/plots/training_data.csv')

# Analyze
print(df.describe())
print(f"Best episode: {df.loc[df['reward'].idxmax()]}")

# Custom plot
import matplotlib.pyplot as plt
plt.plot(df['timestep'], df['reward'])
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('Custom Training Progress')
plt.show()
```

### Load NumPy Data (if pandas not available)
```python
import numpy as np

# Load training data
data = np.load('checkpoints/ppo_minitaur_obstacle/plots/training_data.npz')
episodes = data['episodes']
rewards = data['rewards']
lengths = data['lengths']
timesteps = data['timesteps']

# Analyze
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Best reward: {np.max(rewards):.2f}")
```

## Tips for Better Plots

1. **Train longer**: More episodes = better trends
2. **Save frequently**: Lower `plot_freq` for more snapshots
3. **Monitor during training**: Check plots periodically to catch issues early
4. **Compare runs**: Save different runs to different directories and compare

## Support

If you encounter issues:
1. Check this guide first
2. Run `test_plotting.py` to verify your setup
3. Check console output for error messages
4. Ensure all dependencies are installed:
   ```bash
   pip install matplotlib numpy pandas Pillow
   ```

Happy training! 🤖📊
