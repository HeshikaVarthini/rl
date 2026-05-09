import os
import logging
import time
from datetime import datetime
import sys

class LoggerManager:
    """
    Class for managing multiple loggers
    Records different content in separate files
    """
    
    def __init__(self, log_dir=None):
        """
        Initialize the logger manager
        
        Args:
            log_dir (str, optional): Directory to save log files
                                     Uses 'logs/YYYYMMDD-HHMMSS' if not specified
        """
        # Set log directory
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join("logs", timestamp)
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Logger dictionary
        self.loggers = {}
        
        # Configure root logger
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            # Clear root logger handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
    
    def get_logger(self, name, log_file=None, level=logging.INFO, console_output=True):
        """
        Get or create a logger with the specified name
        
        Args:
            name (str): Logger name
            log_file (str, optional): Log filename. Uses 'name.log' if None
            level (int): Log level
            console_output (bool): Whether to also output to console
            
        Returns:
            logging.Logger: Configured logger
        """
        # Return existing logger if already created
        if name in self.loggers:
            return self.loggers[name]
        
        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Stop propagation to parent loggers
        
        # Clear existing handlers if any
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Set log file path
        if log_file is None:
            log_file = f"{name}.log"
        log_path = os.path.join(self.log_dir, log_file)
        
        # Set file handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        
        # Set formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add console handler if needed
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add to logger dictionary
        self.loggers[name] = logger
        
        return logger