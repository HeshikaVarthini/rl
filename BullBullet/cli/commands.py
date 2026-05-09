import os
import logging
from env.simulation_runner import SimulationRunner
from utils.logging_setup import setup_logging

def execute_command(args):
    """
    Execute appropriate command based on command line arguments
    
    Args:
        args: Command line arguments parsed by argparse
    """
    # Logging setup
    console_level = logging.INFO
    logger_manager = setup_logging(console_level, args.quiet)
    main_logger = logger_manager.get_logger('main')
    
    try:
        # Verify config file exists
        config_path = _validate_config_file(args, main_logger)
        
        # Execute based on selected mode
        if args.mode == 'sim':
            _run_simulation_mode(args, main_logger, config_path, logger_manager)
        elif args.mode == 'train':
            _run_training_mode(args, main_logger, config_path)
        elif args.mode == 'eval':
            _run_evaluation_mode(args, main_logger, config_path)
            
    except Exception as e:
        _handle_error(e, args, main_logger)
        return 1
    
    if not args.quiet:
        main_logger.info("Program terminating")
    
    return 0

def _validate_config_file(args, logger):
    """Verify config file exists and normalize path"""
    config_path = args.config
    
    # If not absolute path and doesn't start with configs/, automatically add it
    if not os.path.isabs(config_path) and not config_path.startswith('configs/'):
        config_path = os.path.join('configs', config_path)
        logger.info(f"Adjusted config file path to '{config_path}'")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        if not args.quiet:
            logger.warning(f"Config file '{config_path}' not found. Using default settings.")
        return None
    else:
        if not args.quiet:
            logger.info(f"Loading from config file '{config_path}'")
        return config_path

def _run_simulation_mode(args, logger, config_path, logger_manager):
    """Run normal simulation mode"""
    if not args.quiet:
        logger.info("Running in normal simulation mode")
    
    runner = SimulationRunner(config_path, logger_manager, console_output=not args.quiet)
    runner.run()
    
    if not args.quiet:
        logger.info("Simulation execution complete")

def _run_training_mode(args, logger, config_path):
    """Run reinforcement learning training mode"""
    if not args.quiet:
        logger.info(f"Starting reinforcement learning training - Epochs: {args.epochs}")
    
    # Import RL components (lazy import to reduce startup time)
    from rl.trainer import RLTrainer
    
    # Initialize RL trainer
    trainer = RLTrainer(
        config_file=config_path,
        render=args.render,
        console_output=not args.quiet
    )
    
    # Run training
    model_path = trainer.train(epochs=args.epochs)
    
    # Display training completion info (important enough to show even with quiet mode)
    print(f"Training complete - Model saved to: {model_path}")
    logger.info(f"Training complete - Model saved to: {model_path}")

def _run_evaluation_mode(args, logger, config_path):
    """Run reinforcement learning evaluation mode"""
    if args.model is None:
        print("Error: Evaluation mode requires model path (--model)")
        logger.error("Evaluation mode requires model path (--model)")
        raise ValueError("Model path required for evaluation mode")
    
    if not args.quiet:
        logger.info(f"Starting model evaluation - Model: {args.model}")
    
    # Import RL components (lazy import to reduce startup time)
    from rl.trainer import RLTrainer
    
    # Initialize RL trainer
    trainer = RLTrainer(
        config_file=config_path,
        console_output=not args.quiet
    )
    
    # Run evaluation
    reward, success_rate = trainer.evaluate(model_path=args.model)
    
    # Display evaluation results (important enough to show even with quiet mode)
    print(f"Evaluation complete - Average reward: {reward:.2f}, Success rate: {success_rate:.2f}%")
    logger.info(f"Evaluation complete - Average reward: {reward:.2f}, Success rate: {success_rate:.2f}%")

def _handle_error(error, args, logger):
    """Error handling"""
    print(f"Error: An error occurred during execution: {error}")
    logger.error(f"An error occurred during execution: {error}")