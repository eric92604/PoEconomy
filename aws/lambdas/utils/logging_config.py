#!/usr/bin/env python3
"""
Logging configuration for Lambda functions.

This module provides standardized logging setup for AWS Lambda functions.
"""

import logging
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class MLLogger:
    """Logger for ML operations with structured logging."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        structured_logging: bool = True
    ):
        """
        Initialize ML logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            console_output: Whether to output to console
            structured_logging: Whether to use structured logging format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Reset handlers
        self.logger.handlers.clear()
        
        # Setup formatter
        if structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Setup console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Setup file handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.start_time = time.time()
        self.operation_times: Dict[str, float] = {}
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        if extra:
            self.logger.info(f"{message} | Extra: {extra}")
        else:
            self.logger.info(message)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        if extra:
            self.logger.debug(f"{message} | Extra: {extra}")
        else:
            self.logger.debug(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        if extra:
            self.logger.warning(f"{message} | Extra: {extra}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        if extra:
            self.logger.error(f"{message} | Extra: {extra}")
        else:
            self.logger.error(message)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        if extra:
            self.logger.critical(f"{message} | Extra: {extra}")
        else:
            self.logger.critical(message)
    
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.operation_times[operation_name] = time.time()
        self.logger.info(f"Started operation: {operation_name}")
    
    def end_operation(self, operation_name: str) -> None:
        """End timing an operation and log duration."""
        if operation_name in self.operation_times:
            duration = time.time() - self.operation_times[operation_name]
            self.logger.info(f"Completed operation: {operation_name} (took {duration:.2f}s)")
            del self.operation_times[operation_name]
        else:
            self.logger.warning(f"Operation {operation_name} was not started")
    
    def get_operation_summary(self) -> Dict[str, float]:
        """Get summary of operation times."""
        return self.operation_times.copy()


def setup_standard_logging(
    name: str,
    level: str = "INFO",
    console_output: bool = True,
    suppress_external: bool = True
) -> logging.Logger:
    """
    Setup standard Python logging for non-ML components.
    
    This function provides standardized logging for non-ML components (AWS Lambda,
    entrypoints, etc.) with:
    - Consistent formatting with function names and line numbers
    - Automatic external library suppression
    - Console output support
    - No file logging (suitable for serverless environments)
    
    Args:
        name: Logger name (should be descriptive, e.g., "IngestionHandler")
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to output to console
        suppress_external: Whether to suppress external library logs
        
    Returns:
        Configured standard Python logger
        
    Example:
        logger = setup_standard_logging(
            name="LambdaHandler",
            level="INFO"
        )
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Setup formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Suppress external library logs
    if suppress_external:
        _suppress_external_libraries()
    
    return logger


def _suppress_external_libraries() -> None:
    """Suppress noisy external library logs with standardized levels."""
    external_loggers = {
        'boto3': logging.WARNING,
        'botocore': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'matplotlib': logging.WARNING,
        'numpy': logging.WARNING,
        'pandas': logging.WARNING,
        'sklearn': logging.WARNING,
        'lightgbm': logging.WARNING,
        'optuna': logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


__all__ = [
    "MLLogger",
    "setup_standard_logging"
]

