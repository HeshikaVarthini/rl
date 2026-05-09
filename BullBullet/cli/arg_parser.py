import argparse

def create_parser():
    """Creates and configures the command line argument parser"""
    parser = argparse.ArgumentParser(description='Quadruped Robot Simulation')
    
    # Add common options to the main parser
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    # Configure subcommands
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Execution mode')
    
    # Simulation mode
    sim_parser = subparsers.add_parser('sim', help='Normal simulation execution')
    
    # Reinforcement learning training mode
    train_parser = subparsers.add_parser('train', help='Reinforcement learning training')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--render', action='store_true', help='Render training process')
    
    # Reinforcement learning evaluation mode
    eval_parser = subparsers.add_parser('eval', help='Evaluate reinforcement learning model')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to model for evaluation')
    
    # Note: The common options are already defined in the main parser and will be inherited by subparsers
    # The following code is redundant and can be removed:
    #
    # for p in [sim_parser, train_parser, eval_parser]:
    #     p.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file')
    #     p.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    return parser