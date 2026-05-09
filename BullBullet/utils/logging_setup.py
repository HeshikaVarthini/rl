import logging
import sys
from utils.logger import LoggerManager

def setup_logging(console_level=logging.INFO, quiet=False):
    """
    Initial setup for logging
    
    Args:
        console_level: Log level for console output
        quiet: If True, suppress general log output to console
        
    Returns:
        LoggerManager: Logger manager instance
    """
    # Initialize logger manager
    logger_manager = LoggerManager()
    
    # Get main logger (no console output if quiet=True)
    main_logger = logger_manager.get_logger('main', console_output=not quiet, level=console_level)
    
    # Log the first message
    if not quiet:
        main_logger.info("Starting quadruped robot simulation")
    
    # Initialize other necessary loggers
    sim_logger = logger_manager.get_logger('simulation', console_output=False)
    env_logger = logger_manager.get_logger('environment', console_output=False)
    rl_logger = logger_manager.get_logger('rl', console_output=False)
    
    return logger_manager