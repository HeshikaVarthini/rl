@echo off
REM Quick Start Script for Training with Plotting
REM This script tests plotting, runs training, and shows results

echo ============================================================
echo Minitaur Obstacle Avoidance Training - Quick Start
echo ============================================================
echo.

echo Step 1: Testing matplotlib plotting...
echo --------------------------------------------------------
python test_plotting.py
echo.

echo Step 2: Starting training...
echo --------------------------------------------------------
echo Press Ctrl+C to stop training early
echo Plots will be saved to: checkpoints\ppo_minitaur_obstacle\plots
echo.
python train_minitaur_obstacle_rl.py
echo.

echo Step 3: Viewing training plots...
echo --------------------------------------------------------
python view_training_plots.py
echo.

echo ============================================================
echo Training Complete!
echo ============================================================
echo Check the following locations:
echo   - Plots: checkpoints\ppo_minitaur_obstacle\plots\
echo   - Models: checkpoints\ppo_minitaur_obstacle\
echo   - TensorBoard: tensorboard --logdir=checkpoints\ppo_minitaur_obstacle\tb
echo ============================================================
pause
