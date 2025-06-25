#!/usr/bin/env python3
"""
Parallel Currency Model Training for Google Cloud c4d VMs

This script demonstrates the optimized parallel training pipeline that maximizes
CPU utilization on c4d instances (8 vCPU, 4 core with hyperthreading).

Features:
- Parallel currency training (4 workers)
- Parallel Optuna optimization (2-3 workers per currency)
- CPU utilization monitoring
- Memory usage tracking
- Performance metrics logging

Usage:
    python train_models_parallel.py --config c4d_optimized
    python train_models_parallel.py --config c4d_high_performance
    python train_models_parallel.py --monitor-resources
"""

import sys
import os
import argparse
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import multiprocessing as mp
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.model_training_pipeline import ModelTrainingPipeline
from config.training_config import (
    MLConfig, ModelConfig, DataConfig, ProcessingConfig, 
    PathConfig, LoggingConfig, ExperimentConfig,
    get_production_config, get_development_config, get_test_config
)
from utils.logging_utils import setup_ml_logging, MLLogger


class ResourceMonitor:
    """Monitor CPU and memory usage during training."""
    
    def __init__(self, logger: MLLogger, interval: float = 5.0):
        """
        Initialize resource monitor.
        
        Args:
            logger: Logger instance
            interval: Monitoring interval in seconds
        """
        self.logger = logger
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'cpu_per_core': [],
            'max_cpu': 0.0,
            'max_memory': 0.0,
            'avg_cpu': 0.0,
            'avg_memory': 0.0
        }
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring and calculate statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate statistics
        if self.stats['cpu_usage']:
            self.stats['avg_cpu'] = sum(self.stats['cpu_usage']) / len(self.stats['cpu_usage'])
            self.stats['max_cpu'] = max(self.stats['cpu_usage'])
        
        if self.stats['memory_usage']:
            self.stats['avg_memory'] = sum(self.stats['memory_usage']) / len(self.stats['memory_usage'])
            self.stats['max_memory'] = max(self.stats['memory_usage'])
        
        self.logger.info("Stopped resource monitoring")
        self._log_final_stats()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Store statistics
                self.stats['cpu_usage'].append(cpu_percent)
                self.stats['memory_usage'].append(memory_percent)
                self.stats['cpu_per_core'].append(cpu_per_core)
                
                # Log current usage
                core_usage = ', '.join([f'Core{i}: {usage:.1f}%' for i, usage in enumerate(cpu_per_core)])
                self.logger.info(
                    f"Resource Usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}% "
                    f"({memory.used // (1024**3):.1f}GB/{memory.total // (1024**3):.1f}GB), "
                    f"Per-core: [{core_usage}]"
                )
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                break
    
    def _log_final_stats(self):
        """Log final resource usage statistics."""
        self.logger.info("=== RESOURCE USAGE SUMMARY ===")
        self.logger.info(f"Average CPU Usage: {self.stats['avg_cpu']:.1f}%")
        self.logger.info(f"Maximum CPU Usage: {self.stats['max_cpu']:.1f}%")
        self.logger.info(f"Average Memory Usage: {self.stats['avg_memory']:.1f}%")
        self.logger.info(f"Maximum Memory Usage: {self.stats['max_memory']:.1f}%")
        
        # Per-core utilization analysis
        if self.stats['cpu_per_core']:
            avg_per_core = []
            for core_idx in range(len(self.stats['cpu_per_core'][0])):
                core_usages = [reading[core_idx] for reading in self.stats['cpu_per_core']]
                avg_usage = sum(core_usages) / len(core_usages)
                max_usage = max(core_usages)
                avg_per_core.append(avg_usage)
                self.logger.info(f"Core {core_idx}: Avg {avg_usage:.1f}%, Max {max_usage:.1f}%")
            
            # Check if all cores are being utilized
            cores_active = sum(1 for usage in avg_per_core if usage > 10.0)
            total_cores = len(avg_per_core)
            utilization_efficiency = cores_active / total_cores * 100
            
            self.logger.info(f"Core Utilization: {cores_active}/{total_cores} cores active ({utilization_efficiency:.1f}%)")
            
            if utilization_efficiency < 75:
                self.logger.warning("LOW CORE UTILIZATION DETECTED - Check parallelization settings")
            else:
                self.logger.info("GOOD CORE UTILIZATION - Parallelization working effectively")


def setup_environment_for_c4d():
    """Configure environment variables for optimal c4d VM performance."""
    env_vars = {
        # Thread control for avoiding oversubscription
        'OMP_NUM_THREADS': '2',        # 2 threads per worker
        'MKL_NUM_THREADS': '2',
        'NUMEXPR_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'BLIS_NUM_THREADS': '2',
        'VECLIB_MAXIMUM_THREADS': '2',
        
        # Memory optimizations
        'MALLOC_TRIM_THRESHOLD_': '100000',
        'PYTHONHASHSEED': '42',        # Reproducible results
        
        # ML library optimizations
        'TF_CPP_MIN_LOG_LEVEL': '2',   # Reduce TensorFlow logging
        'CUDA_VISIBLE_DEVICES': '',    # Disable GPU to focus on CPU
    }
    
    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            print(f"Set {var}={value}")


def validate_c4d_configuration():
    """Validate that we're running on a c4d-compatible system."""
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    print(f"System Configuration:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Memory: {memory_gb}GB")
    
    if cpu_count < 8:
        print("WARNING: Expected 8 vCPUs for c4d instance, found", cpu_count)
    
    if memory_gb < 30:
        print("WARNING: Expected ~32GB RAM for c4d instance, found", memory_gb, "GB")
    
    print("System appears compatible with c4d configuration\n")


def get_mode_config(mode: str, **overrides) -> MLConfig:
    """
    Get configuration based on mode with optional overrides.
    
    Args:
        mode: Configuration mode (production, development, test)
        **overrides: Additional configuration overrides
        
    Returns:
        MLConfig instance
    """
    if mode == 'production':
        config = get_production_config()
        # Production optimizations for Google Cloud c4d VMs
        config.model.max_currency_workers = 4
        config.model.max_optuna_workers = 2
        config.model.currency_worker_threads = 2
        config.model.model_n_jobs = 2
        config.model.optuna_trials_per_worker = 100
        config.experiment.description = "Production parallel training"
        
    elif mode == 'development':
        config = get_development_config()
        # Development optimizations for faster iteration
        config.model.max_currency_workers = 2
        config.model.max_optuna_workers = 2
        config.model.currency_worker_threads = 1
        config.model.model_n_jobs = 1
        config.model.optuna_trials_per_worker = 25
        config.experiment.description = "Development parallel training"
        
    elif mode == 'test':
        config = get_test_config()
        # Test optimizations for quick verification
        config.model.max_currency_workers = 2
        config.model.max_optuna_workers = 2
        config.model.currency_worker_threads = 1
        config.model.model_n_jobs = 1
        config.model.optuna_trials_per_worker = 2
        config.model.n_trials = 3
        config.model.cv_folds = 2
        config.data.max_currencies_to_train = 1
        config.experiment.description = "Fast test mode for pipeline verification"
        config.experiment.tags = ["test", "verification", "minimal"]
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'production', 'development', or 'test'")
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.processing, key):
            setattr(config.processing, key, value)
        elif hasattr(config.experiment, key):
            setattr(config.experiment, key, value)
    
    return config


def run_parallel_training(mode: str, monitor_resources: bool = False, 
                         data_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Run parallel training with specified mode.
    
    Args:
        mode: Training mode (production, development, test)
        monitor_resources: Whether to monitor CPU/memory usage
        data_path: Optional path to training data
        **kwargs: Additional configuration overrides
        
    Returns:
        Training results and performance metrics
    """
    # Get configuration based on mode
    config = get_mode_config(mode, **kwargs)
    
    # Create pipeline
    pipeline = ModelTrainingPipeline(config)
    
    # Setup resource monitoring
    monitor = None
    if monitor_resources:
        monitor = ResourceMonitor(pipeline.logger, interval=3.0)
        monitor.start_monitoring()
    
    # Log configuration details
    pipeline.logger.info("=== PARALLEL TRAINING CONFIGURATION ===")
    pipeline.logger.info(f"Mode: {mode}")
    pipeline.logger.info(f"Experiment ID: {config.experiment.experiment_id}")
    pipeline.logger.info(f"Max Currency Workers: {config.model.max_currency_workers}")
    pipeline.logger.info(f"Max Optuna Workers: {config.model.max_optuna_workers}")
    pipeline.logger.info(f"Worker Threads: {config.model.currency_worker_threads}")
    pipeline.logger.info(f"Model N-Jobs: {config.model.model_n_jobs}")
    pipeline.logger.info(f"Optuna Trials per Worker: {config.model.optuna_trials_per_worker}")
    pipeline.logger.info(f"Total Optuna Trials: {config.model.n_trials}")
    pipeline.logger.info(f"Train all currencies: {config.data.train_all_currencies}")
    pipeline.logger.info(f"Min avg value threshold: {config.data.min_avg_value_threshold}")
    pipeline.logger.info("Parallel processing: ENABLED (default)")
    pipeline.logger.info("=" * 45)
    
    # Run training
    start_time = time.time()
    try:
        results = pipeline.train_all_currencies(data_path)
        
        training_time = time.time() - start_time
        pipeline.logger.info(f"Total training time: {training_time:.2f} seconds")
        
        # Stop monitoring and get stats
        if monitor:
            monitor.stop_monitoring()
        
        return {
            'results': results,
            'training_time': training_time,
            'resource_stats': monitor.stats if monitor else None,
            'mode': mode
        }
        
    except Exception as e:
        if monitor:
            monitor.stop_monitoring()
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parallel Model Training with Resource Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production training with monitoring
  python train_models_parallel.py --mode production --monitor-resources
  
  # Development training (faster)
  python train_models_parallel.py --mode development --monitor-resources
  
  # Quick test (30 seconds)
  python train_models_parallel.py --mode test --monitor-resources
  
  # Train ALL currencies with sufficient data (regardless of value)
  python train_models_parallel.py --train-all-currencies
  
  # Train ALL currencies with custom thresholds
  python train_models_parallel.py --train-all-currencies --min-avg-value 0.1 --min-records 50
  
  # Custom data file
  python train_models_parallel.py --data-path /path/to/data.parquet
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'test'],
        default='production',
        help='Training mode (default: production)'
    )
    
    parser.add_argument(
        '--monitor-resources', 
        action='store_true',
        help='Monitor CPU and memory usage during training'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--validate-system',
        action='store_true',
        help='Validate system configuration for c4d compatibility'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help='Custom experiment ID'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        help='Experiment description'
    )
    
    parser.add_argument(
        '--tags',
        nargs='*',
        default=[],
        help='Experiment tags'
    )
    
    parser.add_argument(
        '--train-all-currencies',
        action='store_true',
        help='Train models for ALL currencies with sufficient data (not just high-value ones)'
    )
    
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup environment
    setup_environment_for_c4d()
    
    # Validate system if requested
    if args.validate_system:
        validate_c4d_configuration()
    
    # Prepare configuration overrides
    overrides = {}
    if args.experiment_id:
        overrides['experiment_id'] = args.experiment_id
    if args.description:
        overrides['description'] = args.description
    if args.tags:
        overrides['tags'] = args.tags
    if args.train_all_currencies:
        overrides['train_all_currencies'] = True
        # When training all currencies, set min value to 0.1 by default
        # This ensures we get ALL currencies that have any meaningful data
        if not args.min_avg_value:
            overrides['min_avg_value_threshold'] = 0.1
    if args.min_avg_value is not None:
        overrides['min_avg_value_threshold'] = args.min_avg_value
    if args.min_records:
        overrides['min_records_threshold'] = args.min_records
    
    print(f"Starting parallel training in {args.mode} mode...")
    print(f"Resource monitoring: {'enabled' if args.monitor_resources else 'disabled'}")
    if args.train_all_currencies:
        print("Training ALL currencies with sufficient data")
    print("=" * 60)
    
    try:
        results = run_parallel_training(
            mode=args.mode,
            monitor_resources=args.monitor_resources,
            data_path=args.data_path,
            **overrides
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Total models trained: {len(results['results'])}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        
        if results['resource_stats']:
            stats = results['resource_stats']
            print(f"Average CPU usage: {stats['avg_cpu']:.1f}%")
            print(f"Maximum CPU usage: {stats['max_cpu']:.1f}%")
            print(f"Average memory usage: {stats['avg_memory']:.1f}%")
            
            # Check if parallelization was effective
            if stats['max_cpu'] > 70:
                print("✅ HIGH CPU UTILIZATION - Parallelization effective!")
            elif stats['max_cpu'] > 40:
                print("⚠️  MODERATE CPU UTILIZATION - Room for improvement")
            else:
                print("❌ LOW CPU UTILIZATION - Check parallelization settings")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 