# RL Quadruped (Minitaur)

**PyBullet simulation → PPO training → obstacle-aware locomotion**

End-to-end reinforcement learning for a quadruped robot with proximity sensing: train with Stable-Baselines3 (PPO), save checkpoints and TensorBoard logs, then evaluate or record demos.

## Demo

<p align="center">
  <video src="https://github.com/HeshikaVarthini/rl/raw/main/DEMO2.mp4" controls width="90%">
  </video>
</p>

<p align="center">
  <a href="https://github.com/HeshikaVarthini/rl/raw/main/DEMO2.mp4">Download / open <code>DEMO2.mp4</code></a>
  ·
  <a href="https://github.com/HeshikaVarthini">Profile</a>
</p>

## Architecture

```
  PyBullet (Minitaur + sensors + obstacles)
           │
           ▼
  Gym env + reward shaping + action wrappers
           │
           ▼
  PPO (Stable-Baselines3)  ──►  checkpoints/*.zip, plots, TensorBoard
           │
           ▼
  Evaluation / screen recording  ──►  policies (.pth / .zip), DEMO videos
```

## Features

- **Physics simulation** with PyBullet and a custom **Minitaur with sensors** (`minisenchum1.py`)
- **PPO training** with checkpointing, training curves, and **live plot PNGs** under `checkpoints/ppo_minitaur_obstacle/plots/`
- **Reward shaping** for forward progress, obstacle proximity, and stable gait (`train_minitaur_obstacle_rl.py`, `runrlmain.py`)
- **Pretrained artifacts** in-repo: policy zips, `.pth` weights, logs, and TensorBoard event files for reproducibility
- **Quick start** batch script for train + plot workflow (`quick_start.bat`)

## Setup

### 1. Python environment

```powershell
cd "path\to\rl\env"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train (PPO + obstacle avoidance)

```powershell
python train_minitaur_obstacle_rl.py
```

Outputs land under `checkpoints\ppo_minitaur_obstacle\` (models, `plots\`, `tb\`).

### 3. Evaluate a saved model

```powershell
python eval_minitaur_obstacle_rl.py
```

Point `eval_minitaur_obstacle_rl.py` at your `.zip` checkpoint if it differs from the default path inside the script.

### 4. One-shot train + plots (Windows)

```powershell
.\quick_start.bat
```

### 5. TensorBoard (optional)

```powershell
tensorboard --logdir checkpoints\ppo_minitaur_obstacle\tb
```

## Project layout (high level)

```
env/
├── README.md
├── requirements.txt
├── DEMO2.mp4                 # demo recording (this repo)
├── minisenchum1.py           # Minitaur + sensors environment
├── train_minitaur_obstacle_rl.py
├── runrlmain.py              # training entry with wrappers / config
├── eval_minitaur_obstacle_rl.py
├── checkpoints/              # PPO zips, plots, TensorBoard
├── logs/                     # additional run logs
├── BullBullet/               # related RL / sim utilities
└── ...
```

## Repository

This project is hosted at **[github.com/HeshikaVarthini/rl](https://github.com/HeshikaVarthini/rl)**.

## Note on assets

Checkpoints, logs, and media files are included so results stay next to the code. **Do not commit secrets** (API keys, personal paths); keep those in local-only files or environment variables.
