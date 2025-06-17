"""
Comprehensive logging utilities for ML operations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import traceback
from contextlib import contextmanager
from functools import wraps
import time
import pandas as pd
import numpy as np


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
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
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
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        if structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.start_time = time.time()
        self.operation_times = {}
    
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
    def log_operation(self, operation_name: str):
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
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        experiment_id: Experiment ID for log file naming
        console_output: Whether to output to console
        suppress_external: Whether to suppress external library logs
        
    Returns:
        Configured MLLogger instance
    """
    # Determine log file path
    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        if experiment_id:
            log_file = str(log_dir_path / f"{name}_{experiment_id}.log")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(log_dir_path / f"{name}_{timestamp}.log")
    
    # Create logger
    ml_logger = MLLogger(
        name=name,
        level=level,
        log_file=log_file,
        console_output=console_output
    )
    
    # Suppress external library logs if requested
    if suppress_external:
        logging.getLogger('lightgbm').setLevel(logging.WARNING)
        logging.getLogger('xgboost').setLevel(logging.WARNING)
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('optuna').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return ml_logger


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