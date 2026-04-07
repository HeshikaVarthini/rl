# Troubleshooting: No Plots Generated

## Problem
You ran training but no plots were generated in the `plots/` directory, and the CSV file is empty.

## Most Likely Causes

### 1. **No Episodes Have Completed Yet** ⭐ (Most Common)
**Symptoms:**
- Training is running
- No "Episode X: Reward=..." messages in console
- CSV file exists but is empty or has only headers
- No PNG files

**Why This Happens:**
- Episodes need to **complete** (reach `done=True`) before they're recorded
- If episodes are very long, you might not see any completed episodes for a while
- The robot might be stuck or episodes might have very high step limits

**Solution:**
```bash
# Run the diagnostic tool
python diagnose_plotting.py

# This will tell you:
# - If training has run
# - If episodes are completing
# - What might be wrong
```

### 2. **Training Hasn't Run Long Enough**
**Symptoms:**
- Only ran training for a few hundred/thousand steps
- No model checkpoints yet
- No console output about episodes

**Solution:**
- Let training run for at least **10,000 timesteps**
- Watch for "Episode X: Reward=..." messages
- First plot will be generated after 2,000 timesteps (if episodes complete)

### 3. **Callback Not Initialized**
**Symptoms:**
- No `plots/` directory created
- No console message about "Callback initialized"

**Solution:**
```bash
# Test the callback system
python test_callback.py

# This runs a short training session to verify callbacks work
```

## Quick Diagnostic Steps

### Step 1: Run Diagnostic Tool
```bash
python diagnose_plotting.py
```

This will check:
- ✓ Directory structure
- ✓ File existence
- ✓ Training evidence
- ✓ Matplotlib functionality

### Step 2: Check Console Output
When training runs, you should see:
```
Callback initialized. Plots will be saved every 2,000 timesteps.
Plot directory: checkpoints/ppo_minitaur_obstacle/plots
Starting training...
...
Episode 1: Reward=123.45, Length=234
Episode 2: Reward=145.67, Length=189
...
Training plots saved to: checkpoints/ppo_minitaur_obstacle/plots/training_progress_2000.png
```

**If you DON'T see "Episode X:" messages:**
- Episodes aren't completing
- Training might need to run longer
- Environment might have issues

### Step 3: Check Episode Completion
Your episodes complete when:
- Robot collides with obstacle (`done=True`)
- Episode reaches max steps
- Robot falls or fails

**To check if episodes are completing:**
1. Look for "Episode X:" messages in console
2. Check if `training_data.csv` has data rows
3. Run `diagnose_plotting.py`

## Solutions

### Solution 1: Wait for Episodes to Complete
```bash
# Let training run longer
python train_minitaur_obstacle_rl.py

# Wait for console messages like:
# Episode 1: Reward=123.45, Length=234
```

### Solution 2: Test with Short Episodes
Create a test script with shorter episodes:

```python
# test_short_episodes.py
import os
os.environ['TOTAL_STEPS'] = '20000'  # Run longer
python train_minitaur_obstacle_rl.py
```

### Solution 3: Check Environment
Make sure your environment is working:

```python
# test_env.py
from minisenchum1 import MinitaurWithSensors

env = MinitaurWithSensors(render=False)
obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Episode finished at step {i}")
        print(f"Reward: {reward}")
        break
else:
    print("Episode didn't finish in 1000 steps")

env.close()
```

### Solution 4: Force Plot Generation
Modify the callback to plot more frequently:

In `train_minitaur_obstacle_rl.py`, change:
```python
plot_freq=2_000,  # Change to 500 or 1000 for more frequent plots
```

## Understanding the Flow

```
Training Starts
    ↓
Callback Initialized
    ↓
Episodes Run
    ↓
Episode Completes (done=True)  ← YOU ARE HERE (probably stuck)
    ↓
Episode Recorded
    ↓
After 2000 timesteps → Plot Generated
    ↓
Plot Saved to PNG
    ↓
Data Saved to CSV
```

**If plots aren't generating, you're likely stuck at "Episode Completes"**

## Verification Checklist

Run through this checklist:

- [ ] Training script runs without errors
- [ ] `checkpoints/ppo_minitaur_obstacle/plots/` directory exists
- [ ] Console shows "Callback initialized" message
- [ ] Console shows "Episode X:" messages ← **CRITICAL**
- [ ] `training_data.csv` exists
- [ ] CSV file has data rows (not just header)
- [ ] PNG files exist in plots directory

**If any checkbox is unchecked, that's where the problem is!**

## Common Scenarios

### Scenario A: Training Just Started
**What you see:**
- Plots directory exists
- No PNG files
- CSV file empty or doesn't exist
- No "Episode X:" messages yet

**What to do:**
✅ **This is normal!** Wait for first episode to complete.

### Scenario B: Training Running for a While
**What you see:**
- Training running for 5000+ timesteps
- Still no "Episode X:" messages
- No plots

**What to do:**
⚠️ Episodes might be too long. Check:
1. Are episodes completing?
2. Is the robot stuck?
3. Is max episode length too high?

### Scenario C: Episodes Completing
**What you see:**
- "Episode X:" messages in console
- But still no plots

**What to do:**
🔧 Run `diagnose_plotting.py` to check matplotlib

## Getting Help

If you're still stuck, gather this information:

1. **Console output** (copy the first 50 lines and last 50 lines)
2. **Directory listing:**
   ```bash
   dir checkpoints\ppo_minitaur_obstacle\plots
   ```
3. **Diagnostic output:**
   ```bash
   python diagnose_plotting.py > diagnostic_output.txt
   ```
4. **Training parameters:**
   - How long did you train?
   - How many timesteps?
   - Did you see "Episode X:" messages?

## Quick Fixes

### Fix 1: Force Episode Completion
Reduce max episode length in your environment to ensure episodes complete quickly.

### Fix 2: Increase Verbosity
In `train_minitaur_obstacle_rl.py`:
```python
reward_loss_callback = RewardLossCallback(
    save_dir=save_dir,
    plot_freq=2_000,
    verbose=2  # Change from 1 to 2 for more output
)
```

### Fix 3: Manual Plot Generation
If you have data in CSV but no plots:

```python
# generate_plots_from_csv.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('checkpoints/ppo_minitaur_obstacle/plots/training_data.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['reward'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.grid(True)
plt.savefig('manual_plot.png')
print("Plot saved to manual_plot.png")
```

## Summary

**Most likely issue:** Episodes haven't completed yet.

**Quick check:**
```bash
python diagnose_plotting.py
```

**Solution:** Let training run until you see "Episode X:" messages in console.

**Still stuck?** Run `test_callback.py` to verify the system works.
