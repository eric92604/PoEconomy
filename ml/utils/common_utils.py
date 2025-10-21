"""
Common utilities for ML pipeline.

This module consolidates shared utilities that are used across multiple
parts of the ML pipeline, including argument parsing, configuration
management, logging, and other common functionality.
"""

import argparse
import logging
import sys
import json
import traceback
import time
from contextlib import contextmanager
from typing import Generator
from typing import List, Optional, Any, Dict
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from ml.config.training_config import MLConfig, get_default_config, get_all_currencies_config, get_high_value_config


def create_base_parser(description: str, epilog: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.
    
    Args:
        description: Description for the argument parser
        epilog: Optional epilog text
        
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    # Common arguments
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'test'],
        default='production',
        help='Training mode (default: production)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Custom experiment ID'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Experiment description'
    )
    
    parser.add_argument(
        '--tags',
        nargs='*',
        default=[],
        help='Experiment tags'
    )
    
    return parser


def add_currency_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add currency-specific arguments to a parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        Updated ArgumentParser
    """
    parser.add_argument(
        '--min-avg-value',
        type=float,
        help='Minimum average value (in Chaos Orbs) for currency inclusion'
    )
    
    parser.add_argument(
        '--min-records',
        type=int,
        help='Minimum number of historical records required'
    )
    
    parser.add_argument(
        '--currencies',
        nargs='*',
        help='Specific currencies to train (space-separated list)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers for training'
    )
    
    parser.add_argument(
        '--max-currencies',
        type=int,
        help='Maximum number of currencies to process (0 = no limit)'
    )
    
    return parser


def add_feature_engineering_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add feature engineering specific arguments to a parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        Updated ArgumentParser
    """
    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help='Skip feature engineering step'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing'
    )
    
    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual currency datasets'
    )
    
    parser.add_argument(
        '--train-all-currencies',
        action='store_true',
        help='Train models for all currencies with sufficient data'
    )
    
    parser.add_argument(
        '--max-league-days',
        type=int,
        help='Maximum days into each league to consider'
    )
    
    return parser


def load_config_from_args(args: Any) -> MLConfig:
    """
    Load configuration based on command line arguments.
    
    Args:
        args: Parsed command line arguments with mode, config, etc.
        
    Returns:
        Loaded MLConfig instance
    """
    # Load from file if specified
    if hasattr(args, 'config') and args.config:
        return MLConfig.from_file(args.config)
    
    # Load based on mode
    if hasattr(args, 'mode'):
        if args.mode == 'production':
            config = get_default_config()
        elif args.mode == 'development':
            config = get_default_config()
        else:  # test
            config = get_default_config()
    else:
        # Default to production if no mode specified
        config = get_default_config()
    
    return config


def apply_args_to_config(config: MLConfig, args: Any) -> MLConfig:
    """
    Apply command line arguments to configuration.
    
    Args:
        config: Base configuration
        args: Parsed command line arguments
        
    Returns:
        Updated configuration
    """
    # Apply common arguments
    if hasattr(args, 'experiment_id') and args.experiment_id:
        config.experiment.experiment_id = args.experiment_id
    
    if hasattr(args, 'description') and args.description:
        config.experiment.description = args.description
    
    if hasattr(args, 'tags') and args.tags:
        config.experiment.tags.extend(args.tags)
    
    # Apply currency-specific arguments
    if hasattr(args, 'min_records') and args.min_records:
        config.data.min_records_threshold = args.min_records
    
    if hasattr(args, 'min_avg_value') and args.min_avg_value is not None:
        config.data.min_avg_value_threshold = args.min_avg_value
    
    if hasattr(args, 'currencies') and args.currencies:
        config.data.max_currencies_to_train = len(args.currencies)
        config.pipeline.currencies_to_train = args.currencies
    
    if hasattr(args, 'max_currencies') and args.max_currencies is not None:
        config.data.max_currencies_to_train = args.max_currencies
    
    if hasattr(args, 'max_workers') and args.max_workers:
        config.model.max_currency_workers = args.max_workers
    
    # Apply feature engineering specific arguments
    if hasattr(args, 'train_all_currencies') and args.train_all_currencies:
        config.data.train_all_currencies = True
        # When training all currencies, set min value to 0.1 by default
        if not hasattr(args, 'min_avg_value') or args.min_avg_value is None:
            config.data.min_avg_value_threshold = 0.1
    
    if hasattr(args, 'parallel') and args.parallel:
        config.processing.use_parallel_processing = True
    
    if hasattr(args, 'save_individual') and args.save_individual:
        config.experiment.save_individual_datasets = True
    
    if hasattr(args, 'max_league_days') and args.max_league_days:
        config.data.max_league_days = args.max_league_days
    
    # Add mode tag
    if hasattr(args, 'mode'):
        config.experiment.tags.append(args.mode)
    
    return config


# ------------------------------------------------------------------
# Logging Utilities
# ------------------------------------------------------------------

def make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON serializable format using industry best practices.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, Path):  # type: ignore[unreachable]
        return str(obj)
    else:
        # Handle pandas NA values and other edge cases
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj


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
        """Log info message with optional structured data."""
        if extra:
            safe_extra = make_json_serializable(extra)
            message = f"{message} | {json.dumps(safe_extra)}"
        self.logger.info(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with optional structured data."""
        if extra:
            safe_extra = make_json_serializable(extra)
            message = f"{message} | {json.dumps(safe_extra)}"
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
        if extra:
            safe_extra = make_json_serializable(extra)
            message = f"{message} | {json.dumps(safe_extra)}"
        
        self.logger.error(message)
        
        if exception:
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with optional structured data."""
        if extra:
            safe_extra = make_json_serializable(extra)
            message = f"{message} | {json.dumps(safe_extra)}"
        self.logger.debug(message)
    
    def log_experiment_start(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """Log experiment start with configuration."""
        extra_data = {
            "experiment_id": experiment_id,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"Starting experiment: {experiment_id}", extra=extra_data)
    
    def log_experiment_end(self, experiment_id: str, results: Dict[str, Any]) -> None:
        """Log experiment end with results."""
        elapsed_time = time.time() - self.start_time
        extra_data = {
            "experiment_id": experiment_id,
            "results": results,
            "elapsed_time_seconds": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"Completed experiment: {experiment_id}", extra=extra_data)
    
    def log_model_training_start(self, currency_pair: str, model_type: str, data_shape: tuple) -> None:
        """Log model training start."""
        extra_data = {
            "currency_pair": currency_pair,
            "model_type": model_type,
            "data_shape": data_shape,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"Training {model_type} model for {currency_pair}", extra=extra_data)
    
    def log_model_training_end(
        self,
        currency_pair: str,
        model_type: str,
        metrics: Dict[str, float],
        training_time: float
    ) -> None:
        """Log model training completion."""
        extra_data = {
            "currency_pair": currency_pair,
            "model_type": model_type,
            "metrics": metrics,
            "training_time_seconds": training_time,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"Completed training {model_type} model for {currency_pair}", extra=extra_data)
    
    def log_data_processing(self, operation: str, input_shape: tuple, output_shape: tuple) -> None:
        """Log data processing operation."""
        extra_data = {
            "operation": operation,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "timestamp": datetime.now().isoformat()
        }
        self.info(f"Data processing: {operation}", extra=extra_data)
    
    @contextmanager
    def log_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Context manager for logging operation duration."""
        start_time = time.time()
        self.info(f"Starting operation: {operation_name}")
        
        try:
            yield
            elapsed_time = time.time() - start_time
            self.operation_times[operation_name] = elapsed_time
            self.info(
                f"Completed operation: {operation_name}",
                extra={"elapsed_time_seconds": elapsed_time}
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.error(
                f"Failed operation: {operation_name}",
                exception=e,
                extra={"elapsed_time_seconds": elapsed_time}
            )
            raise
    
    def get_operation_summary(self) -> Dict[str, float]:
        """Get summary of operation times."""
        return self.operation_times.copy()


def setup_ml_logging(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    console_output: bool = True,
    suppress_external: bool = True
) -> MLLogger:
    """
    Setup ML logging with best practices.
    
    This function provides standardized logging for ML components with:
    - Structured logging format with function names and line numbers
    - Automatic external library suppression
    - File and console output support
    - Experiment tracking integration
    
    Args:
        name: Logger name (should be descriptive, e.g., "CurrencyTrainer")
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (creates directory if needed)
        experiment_id: Experiment ID for log file naming
        console_output: Whether to output to console
        suppress_external: Whether to suppress external library logs
        
    Returns:
        Configured MLLogger instance with structured logging capabilities
        
    Example:
        logger = setup_ml_logging(
            name="ModelTrainer",
            level="INFO",
            log_dir="/path/to/logs",
            experiment_id="exp_001"
        )
    """
    # Setup log file path
    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        if experiment_id:
            log_file = str(log_dir_path / f"{name}_{experiment_id}.log")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(log_dir_path / f"{name}_{timestamp}.log")
    
    # Initialize logger
    ml_logger = MLLogger(
        name=name,
        level=level,
        log_file=log_file,
        console_output=console_output
    )
    
    # Suppress external library logs
    if suppress_external:
        _suppress_external_libraries()
    
    return ml_logger


def _suppress_external_libraries() -> None:
    """Suppress noisy external library logs with standardized levels."""
    external_loggers = {
        'lightgbm': logging.ERROR,
        'xgboost': logging.ERROR,
        'sklearn': logging.ERROR,
        'optuna': logging.ERROR,
        'matplotlib': logging.ERROR,
        'urllib3': logging.ERROR,
        'botocore': logging.WARNING,
        'boto3': logging.WARNING,
        'pandas': logging.WARNING,
        'numpy': logging.WARNING,
        'joblib': logging.WARNING,
        'requests': logging.WARNING,
        'PIL': logging.WARNING,
        'Pillow': logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


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


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: MLLogger, total_items: int, operation_name: str):
        """
        Initialize progress logger.
        
        Args:
            logger: MLLogger instance
            total_items: Total number of items to process
            operation_name: Name of the operation
        """
        self.logger = logger
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Log intervals
        self.log_interval_items = max(1, total_items // 20)  # Log every 5%
        self.log_interval_time = 30  # Log every 30 seconds minimum
    
    def update(self, items_processed: int = 1) -> None:
        """Update progress and log if necessary."""
        self.processed_items += items_processed
        current_time = time.time()
        
        # Check if we should log
        should_log = (
            self.processed_items % self.log_interval_items == 0 or
            current_time - self.last_log_time >= self.log_interval_time or
            self.processed_items == self.total_items
        )
        
        if should_log:
            progress_pct = (self.processed_items / self.total_items) * 100
            elapsed_time = current_time - self.start_time
            
            if self.processed_items > 0:
                estimated_total_time = elapsed_time * (self.total_items / self.processed_items)
                remaining_time = estimated_total_time - elapsed_time
            else:
                remaining_time = 0
            
            extra_data = {
                "processed": self.processed_items,
                "total": self.total_items,
                "progress_percent": round(progress_pct, 1),
                "elapsed_time_seconds": round(elapsed_time, 1),
                "estimated_remaining_seconds": round(remaining_time, 1)
            }
            self.logger.info(f"Progress: {self.operation_name}", extra=extra_data)
            
            self.last_log_time = current_time
    
    def complete(self) -> None:
        """Mark operation as complete."""
        total_time = time.time() - self.start_time
        extra_data = {
            "total_items": self.total_items,
            "total_time_seconds": round(total_time, 1),
            "items_per_second": round(self.total_items / total_time, 2) if total_time > 0 else 0
        }
        self.logger.info(f"Completed: {self.operation_name}", extra=extra_data)


__all__ = [
    "create_base_parser",
    "add_currency_arguments", 
    "add_feature_engineering_arguments",
    "load_config_from_args",
    "apply_args_to_config",
    "make_json_serializable",
    "MLLogger",
    "setup_ml_logging",
    "ProgressLogger"
]
