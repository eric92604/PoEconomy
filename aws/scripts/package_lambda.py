#!/usr/bin/env python3
"""
Lambda packaging script for PoEconomy.
This script packages Lambda functions with their dependencies and ML modules.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    # This script is in aws/scripts/, so go up two levels
    return Path(__file__).parent.parent.parent


def copy_ml_modules(source_dir: Path, dest_dir: Path) -> bool:
    """Copy ML modules from source to destination."""
    ml_source = source_dir / "ml"
    ml_dest = dest_dir / "ml"
    
    print(f"Copying ML modules from {ml_source} to {ml_dest}")
    
    if not ml_source.exists():
        print(f"Error: ML source directory not found: {ml_source}")
        return False
    
    try:
        if ml_dest.exists():
            shutil.rmtree(ml_dest)
        shutil.copytree(ml_source, ml_dest)
        print(f"Successfully copied ML modules")
        
        # Verify the copy
        if ml_dest.exists():
            print(f"ML destination verified: {ml_dest}")
            print(f"ML modules: {list(ml_dest.iterdir())}")
            return True
        else:
            print("Error: ML destination does not exist after copy")
            return False
            
    except Exception as e:
        print(f"Error copying ML modules: {e}")
        return False


def install_dependencies(requirements_file: Path, target_dir: Path) -> bool:
    """Install Python dependencies to target directory."""
    print(f"Installing dependencies from {requirements_file}")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file),
            "-t", str(target_dir),
            "--quiet"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Dependencies installed successfully")
            return True
        else:
            print(f"Warning: Some dependencies failed to install: {result.stderr}")
            # Try to install basic dependencies
            basic_result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "boto3", "requests",
                "-t", str(target_dir),
                "--quiet"
            ], capture_output=True, text=True)
            
            if basic_result.returncode == 0:
                print("Basic dependencies installed successfully")
                return True
            else:
                print(f"Error installing basic dependencies: {basic_result.stderr}")
                return False
                
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False


def create_zip_package(source_dir: Path, zip_path: Path) -> bool:
    """Create a zip file from the source directory."""
    print(f"Creating zip package: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    # Use forward slashes for zip archive
                    arcname_str = str(arcname).replace(os.sep, '/')
                    zipf.write(file_path, arcname_str)
        
        print(f"Successfully created zip package: {zip_path}")
        return True
        
    except Exception as e:
        print(f"Error creating zip package: {e}")
        return False


def package_lambda(
    lambda_name: str,
    handler_file: str,
    output_zip: str,
    project_root: Optional[Path] = None
) -> bool:
    """Package a Lambda function with its dependencies."""
    
    if project_root is None:
        project_root = get_project_root()
    
    print(f"Packaging Lambda function: {lambda_name}")
    print(f"Handler file: {handler_file}")
    print(f"Output zip: {output_zip}")
    print(f"Project root: {project_root}")
    
    # Determine lambda directory
    if lambda_name in ["ingestion", "league_metadata", "daily_aggregation"]:
        lambda_dir = project_root / "aws" / "lambdas" / "ingestion"
    elif lambda_name == "api":
        lambda_dir = project_root / "aws" / "lambdas" / "api"
    elif lambda_name == "prediction_refresh":
        lambda_dir = project_root / "aws" / "lambdas" / "prediction"
    else:
        lambda_dir = project_root / "aws" / "lambdas"
    
    # Validate required files
    handler_path = lambda_dir / handler_file
    config_path = project_root / "aws" / "lambdas" / "config.py"
    init_path = project_root / "aws" / "lambdas" / "__init__.py"
    requirements_path = lambda_dir / "requirements.txt"
    
    if not handler_path.exists():
        print(f"Error: Handler file not found: {handler_path}")
        return False
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return False
    
    if not init_path.exists():
        print(f"Error: Init file not found: {init_path}")
        return False
    
    if not requirements_path.exists():
        print(f"Error: Requirements file not found: {requirements_path}")
        return False
    
    # Create temporary directory for packaging
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        package_dir = temp_path / lambda_name
        
        print(f"Creating package directory: {package_dir}")
        package_dir.mkdir()
        
        # Copy Lambda source files
        print("Copying Lambda source files...")
        shutil.copy2(handler_path, package_dir / handler_file)
        shutil.copy2(config_path, package_dir / "config.py")
        shutil.copy2(init_path, package_dir / "__init__.py")
        shutil.copy2(requirements_path, package_dir / "requirements.txt")

        # Copy shared Lambda utils (logging_config, etc.) as a top-level package
        utils_source = project_root / "aws" / "lambdas" / "utils"
        utils_dest = package_dir / "utils"
        if utils_source.exists():
            print(f"Copying Lambda utils from {utils_source} to {utils_dest}")
            shutil.copytree(utils_source, utils_dest)
        else:
            print(f"Warning: Lambda utils directory not found: {utils_source}")

        # Copy ML modules
        if not copy_ml_modules(project_root, package_dir):
            print("Warning: Failed to copy ML modules, Lambda may have import errors")
        
        # Install dependencies
        if not install_dependencies(requirements_path, package_dir):
            print("Warning: Failed to install some dependencies")
        
        # Verify package contents
        print("Package contents:")
        for item in package_dir.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(package_dir)}")
        
        # Create zip package
        if not create_zip_package(package_dir, Path(output_zip)):
            return False
        
        print(f"Successfully packaged {lambda_name} Lambda function")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Package PoEconomy Lambda functions")
    parser.add_argument("lambda_name", help="Name of the Lambda function")
    parser.add_argument("handler_file", help="Handler file name")
    parser.add_argument("output_zip", help="Output zip file path")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else None
    
    success = package_lambda(
        args.lambda_name,
        args.handler_file,
        args.output_zip,
        project_root
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
