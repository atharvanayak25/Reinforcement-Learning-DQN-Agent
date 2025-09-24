# Reinforcement Learning Deep Q-Network, Double DQN and Dueling DQN

This project is a comprehensive implementation of a Deep Q-Network (DQN) agent and its powerful variants—Double DQN and Dueling DQN—built from scratch using Python, PyTorch, and the Gymnasium library. The agent is designed to learn and solve classic reinforcement learning environments through trial and error. The framework is highly configurable and allows for easy switching between these architectures to enhance training stability and performance.

Key Features
Built with PyTorch: Utilizes modern, efficient deep learning practices.

Multiple DQN Architectures: Implements Standard DQN, Double DQN, and Dueling DQN, which can be enabled via a configuration file.

Hyperparameter-Driven: All training parameters (learning rate, epsilon decay, memory size, etc.) are managed in a central hyperparameters.yml file for easy experimentation.

Performance Visualization: Automatically generates and saves graphs of the mean rewards and epsilon decay during training, providing clear insight into the agent's learning progress.

Model Checkpointing: Saves the best-performing model during training, so you always have access to your top agent.

## Performance Showcase
The agent was successfully trained on multiple environments, each presenting unique challenges.

## 1. CartPole-v1
A classic control problem where the agent must learn to balance a pole on a cart. The graph below shows a clear and stable learning curve as the agent masters the task.


## 2. FlappyBird-v0
A much more challenging environment with sparse rewards and difficult dynamics. The agent shows signs of learning but struggles with stability, highlighting the importance of robust hyperparameter tuning.

## Installation
To get this project up and running on your local machine, follow these steps.

## Clone the repository:
```bash
git clone [https://github.com/atharvanayak25/Reinforcement-Learning-DQN-Agent.git](https://github.com/atharvanayak25/Reinforcement-Learning-DQN-Agent.git)
cd Reinforcement-Learning-DQN-Agent

```

## Create and activate a Python virtual environment:
```bash
# Create the environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

## How to Use

The agent's behavior is controlled through the agent.py script and the hyperparameters.yml configuration file.

## Training an Agent

To start training, you must provide the name of a hyperparameter set from the .yml file and include the --train flag.

```bash
# Train the agent on the CartPole environment
python agent.py cartpole1 --train

# Train the agent on the Flappy Bird environment
python agent.py flappybird1 --train
```

Training logs, the best model (.pt file), and the final performance graph (.png file) will be saved in the runs/ directory.

## Watching a Trained Agent

To watch a pre-trained agent perform, run the script with the --render flag. Make sure a model file (e.g., runs/cartpole1.pt) already exists from a previous training run.

``` bash
# Watch the trained CartPole agent
python agent.py cartpole1
```

