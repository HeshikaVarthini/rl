# Plotting Fix Summary

## Problem
Graphs were not plotting when running the training script from Windows Command Prompt.

## Root Cause
Matplotlib's default backend requires an interactive display (GUI), which is not available in command prompt environments.

## Solution Applied

### 1. Changed Matplotlib Backend
**File**: `train_minitaur_obstacle_rl.py`

**Before**:
```python
import matplotlib.pyplot as plt
```

**After**:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for command prompt
import matplotlib.pyplot as plt
```

The `'Agg'` backend:
- ✅ Works in command prompt without display
- ✅ Saves plots to PNG files
- ✅ No GUI required
- ✅ Fast and reliable

### 2. Disabled Live Plotting
**Before**: Attempted to create interactive plot windows
**After**: All plots saved to disk automatically

### 3. Added Error Handling
- Graceful fallback if pandas not installed (saves as .npz instead)
- Better error messages
- Verification that plot files are created

## How It Works Now

1. **During Training**:
   - Plots are generated every 5,000 timesteps
   - Saved as PNG files to `checkpoints/ppo_minitaur_obstacle/plots/`
   - Console shows progress messages

2. **After Training**:
   - Final plot automatically saved
   - CSV data exported for analysis
   - All plots available in plots directory

## Files Created

### 1. `train_minitaur_obstacle_rl.py` (Modified)
- Added `matplotlib.use('Agg')` for command prompt compatibility
- Improved callback with better error handling
- Added progress messages

### 2. `test_plotting.py` (New)
- Quick test to verify matplotlib works
- Creates sample plot
- Helps diagnose issues

### 3. `view_training_plots.py` (New)
- View saved plots after training
- Lists all available plots
- Opens latest plot in default image viewer

### 4. `PLOTTING_GUIDE.md` (New)
- Complete guide for using the plotting features
- Troubleshooting tips
- Custom analysis examples

## Testing

To verify the fix works:

```bash
# Step 1: Test matplotlib
python test_plotting.py

# Step 2: Run training (even for a short time)
python train_minitaur_obstacle_rl.py

# Step 3: Check plots directory
dir checkpoints\ppo_minitaur_obstacle\plots

# Step 4: View plots
python view_training_plots.py
```

## Expected Output

### Console Output During Training:
```
Plotting mode: Saving plots to disk (command prompt compatible)
Plots will be saved to: checkpoints/ppo_minitaur_obstacle/plots
Starting training...
Total timesteps: 400,000
Number of environments: 1
Save directory: checkpoints/ppo_minitaur_obstacle
==================================================
...
Training plots saved to: checkpoints/ppo_minitaur_obstacle/plots/training_progress_5000.png
...
```

### Files Created:
```
checkpoints/ppo_minitaur_obstacle/plots/
├── training_progress_5000.png
├── training_progress_10000.png
├── training_progress_15000.png
└── training_data.csv
```

## Verification Checklist

- [x] Matplotlib backend set to 'Agg'
- [x] Live plotting disabled
- [x] Plots saved to disk automatically
- [x] Error handling for missing dependencies
- [x] Console messages show progress
- [x] Test script created
- [x] Viewer script created
- [x] Documentation provided

## Benefits

1. **Works in Command Prompt**: No GUI required
2. **Automatic Saving**: All plots saved automatically
3. **Persistent**: Plots remain after training ends
4. **Shareable**: PNG files can be easily shared
5. **Analyzable**: CSV data for custom analysis
6. **Reliable**: No display-related crashes

## Alternative: TensorBoard

If you prefer real-time monitoring, you can also use TensorBoard:

```bash
# In a separate command prompt window
tensorboard --logdir=checkpoints/ppo_minitaur_obstacle/tb

# Open browser to http://localhost:6006
```

TensorBoard provides:
- Real-time updates during training
- Interactive plots
- More detailed metrics
- Comparison between runs

## Summary

The plotting system now works perfectly in Windows Command Prompt by:
1. Using a non-interactive matplotlib backend ('Agg')
2. Saving all plots to disk automatically
3. Providing helper scripts to view plots
4. Including comprehensive documentation

**Result**: You can now monitor your training progress with beautiful graphs, even when running from command prompt! 📊✅
