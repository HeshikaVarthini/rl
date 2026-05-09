# BullBullet - Agility

![BullBullet Simulation](./image/bullbullet.png "BullBullet Simulation")

BullBullet is a framework for simulating quadruped robots using PyBullet. It allows experimentation with basic robot control tasks such as gait pattern generation, obstacle avoidance, and goal reaching. It also aims to achieve autonomous behavior acquisition using reinforcement learning (PPO algorithm).

_Developers: Aizawa, Otsuka_

## Features

- Support for multiple gait patterns (trot, walk, bound, etc.)
- Environment recognition using LiDAR sensors
- Automatic generation of obstacle courses (three types: simple, high-density, and random)
- Goal achievement functionality
- Customizable environment and robot parameters
- Simple operation from the command line
- Reinforcement learning mode
  - Learning through PPO (Proximal Policy Optimization) algorithm
  - Customizable reward function
  - Separation of training and evaluation
  - Automatic plotting of learning curves
  - Log output

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)

### Steps

1. Clone the repository:

```bash
git clone https://github.com/Aizawa-Shun/BullBullet.git
cd BullBullet
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution

To run normal simulation mode:

```bash
python main.py
```

### Command Line Options

BullBullet supports various command line options:

#### Basic Options

```bash
python main.py --config CONFIG_FILE   # Specify configuration file
python main.py --export-default PATH  # Export default configuration
python main.py --verbose              # Detailed log output
python main.py --console-only         # Output logs to console only
python main.py --quiet                # Suppress console log output
```

#### Simulation Mode

```bash
python main.py sim                    # Run normal simulation (default)
```

#### Reinforcement Learning Mode

```bash
# Training mode
python main.py rl --rl-mode train --epochs 100

# Evaluation mode
python main.py rl --rl-mode evaluate --load-model MODEL_PATH

# Other options
python main.py rl --render            # Render during training
python main.py rl --rl-config FILE    # Reinforcement learning configuration file
```

### Configuration Files

BullBullet provides two types of configuration files:

- `configs/config.yaml`: For normal simulation
- `configs/rl_config.yaml`: For reinforcement learning

To export default configuration:

```bash
python main.py --export-default configs/my_config.yaml
```

### Configuration Structure

Configuration files consist of the following sections:

#### Robot Configuration

```yaml
robot:
  urdf_path: models/urdf/svdog2_2_description/svdog2_2.urdf
  position: [0, 0, 0.08] # Initial position [x, y, z]
  rotation: [0, 0, 135] # Initial orientation (Euler angles) [roll, pitch, yaw]
  max_force: 5.0 # Maximum torque for actuators
```

#### Environment Configuration

```yaml
environment:
  use_gui: true # Whether to use GUI
  camera_follow: true # Whether the camera follows the robot
  gravity: [0, 0, -9.8] # Gravity vector [x, y, z]
  timestep: 0.00416667 # Simulation timestep (1/240)
```

#### LiDAR Configuration

```yaml
lidar:
  enabled: true # Whether to enable LiDAR
  num_rays: 36 # Number of rays
  ray_length: 1.0 # Maximum ray length
  ray_start_length: 0.01 # Ray start distance
  ray_color: [0, 1, 0] # Normal ray color [R, G, B]
  ray_hit_color: [1, 0, 0] # Collision ray color [R, G, B]
```

#### Gait Configuration

```yaml
gait:
  amplitude: 0.25 # Joint angle amplitude
  frequency: 1.5 # Walking cycle frequency
  pattern: trot # Gait pattern (trot, walk, bound)
  turn_direction: 0 # Turn direction (-1.0: left, 0: straight, 1.0: right)
  turn_intensity: 0 # Turn intensity (0.0 - 1.0)
```

#### Obstacle Configuration

```yaml
obstacles:
  enabled: true # Whether to enable obstacles
  course_type: simple # Course type (simple, dense, random)
  length: 5.0 # Course length
```

#### Goal Configuration

```yaml
goal:
  enabled: true # Whether to enable goal
  position: [2.0, 0, 0] # Goal position [x, y, z]
  radius: 0.3 # Goal radius
  color: [0.0, 0.8, 0.0, 0.5] # Goal color [R, G, B, A]
```

#### Simulation Configuration

```yaml
simulation:
  max_steps: 5000 # Maximum simulation steps
  debug_interval: 100 # Debug information display interval
```

## Gait Patterns

BullBullet supports the following gait patterns:

- `trot`: Move diagonal legs simultaneously (default)
- `walk`: Move each leg in sequence
- `bound`: Move front and rear leg pairs simultaneously

Gait patterns can be specified in the configuration file:

```yaml
gait:
  pattern: "trot" # Specify one of 'trot', 'walk', 'bound'
  amplitude: 0.25
  frequency: 1.5
```

## Obstacle Courses

BullBullet can generate multiple types of obstacle courses:

- `simple`: Basic obstacle arrangement
- `dense`: Densely arranged obstacles
- `random`: Randomly placed obstacles

Course type can be specified in the configuration file:

```yaml
obstacles:
  enabled: true
  course_type: "simple" # Specify one of 'simple', 'dense', 'random'
  length: 5.0
```

## Project Structure

```
bull_bullet_dogrun/
├── configs/            # Configuration files
│   ├── config.yaml          # Normal simulation configuration
│   └── rl_config.yaml       # Reinforcement learning configuration
├── env/                # Environment-related classes
│   ├── config_loader.py     # Configuration loader
│   ├── gait.py              # Gait generation
│   ├── goal_marker.py       # Goal marker
│   ├── lidar_sensor.py      # LiDAR sensor
│   ├── obstacle_generator.py # Obstacle generation
│   ├── quad_env.py          # Quadruped environment base class
│   └── simulation_runner.py  # Simulation execution class
├── models/             # Robot models (URDF)
├── rl/                 # Reinforcement learning related
│   ├── evaluate.py          # Evaluation environment
│   ├── ppo_agent.py         # PPO agent implementation
│   ├── rl_environment.py    # Reinforcement learning environment
│   └── trainer.py           # Learning trainer
├── utils/              # Utility functions
│   ├── logger.py            # Logging management
│   └── logging_setup.py     # Logging configuration
├── logs/               # Log output directory
├── results/            # Reinforcement learning results directory
└── main.py             # Main entry point
```

## Reinforcement Learning Mode

BullBullet provides a reinforcement learning mode using the PPO (Proximal Policy Optimization) algorithm. It can automatically learn robot control policies and acquire the ability to reach goals while avoiding obstacles.

### Running Reinforcement Learning

```bash
# Training (new model)
python main.py rl --rl-mode train --epochs 100

# Continue training from existing model
python main.py rl --rl-mode train --load-model PATH --epochs 50

# Evaluation mode
python main.py rl --rl-mode evaluate --load-model PATH
```

### Reinforcement Learning Configuration

Reinforcement learning is configured in `configs/rl_config.yaml`:

```yaml
# Reinforcement learning related configuration (partial excerpt)
obstacles:
  enabled: true
  course_type: random # Random courses are effective for learning
  length: 8.0

goal:
  enabled: true
  position: [2.0, 0, 0]
  radius: 0.4
```

### Learning Results

Training results are saved in the `results/ppo_quadruped_TIMESTAMP/` directory:

- `model_final.pt`: Trained model
- `metrics_final.json`: Learning metrics
- `learning_curves_final.png`: Learning curves
- `evaluation_results.json`: Evaluation results
- `hyperparameters.json`: Hyperparameters

### PPO Algorithm Implementation

BullBullet's PPO implementation has the following features:

- Shared model including actor and critic networks
- Stable learning using GAE (Generalized Advantage Estimation)
- Exploration promotion through entropy regularization
- Learning constraints based on KL divergence
- Action selection considering gait stability

## License

[MIT License](LICENSE)

## Contributing

Please open an issue to discuss before making major changes.

## Author

Shun Aizawa
