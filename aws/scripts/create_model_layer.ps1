# PowerShell Script to Create Lambda Layer with ML Models for Cost Optimization
#
# This script packages trained ML models into Lambda layers, which:
# - Provides persistent storage across invocations (no S3 download costs)
# - Allows models up to 250MB total
# - Shared across multiple Lambda functions
# - Versioned for easy rollbacks

param(
    [string]$LayerName = "poeconomy-ml-models",
    [string]$ModelsDir = "ml\models\currency_production",
    [string]$BuildDir = "layer-build",
    [string]$ZipFile = "ml-models-layer.zip",
    [string]$AwsProfile = "",
    [int]$MaxModels = 50
)

Write-Host "🚀 Building Lambda Layer with ML Models..." -ForegroundColor Green

# Clean and create build directory
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path "$BuildDir\python" -Force | Out-Null

# Copy essential models (prioritize by size/usage)
Write-Host "📦 Copying model files..." -ForegroundColor Yellow

if (Test-Path $ModelsDir) {
    # Get all model files and prioritize important currencies
    $priorityCurrencies = @(
        "Divine Orb", "Chaos Orb", "Exalted Orb", "Ancient Orb", 
        "Chromatic Orb", "Jewellers Orb", "Orb of Fusing",
        "Orb of Alchemy", "Orb of Alteration", "Vaal Orb"
    )
    
    # Copy priority models first
    $copiedCount = 0
    foreach ($currency in $priorityCurrencies) {
        $modelPath = Join-Path $ModelsDir $currency
        if (Test-Path $modelPath) {
            $targetPath = Join-Path "$BuildDir\python" $currency
            Copy-Item -Path $modelPath -Destination $targetPath -Recurse -Force
            $copiedCount++
            Write-Host "✓ Copied priority model: $currency" -ForegroundColor Green
        }
    }
    
    # Copy remaining models up to limit
    $remainingModels = Get-ChildItem $ModelsDir -Directory | Where-Object { $_.Name -notin $priorityCurrencies } | Select-Object -First ($MaxModels - $copiedCount)
    foreach ($model in $remainingModels) {
        $targetPath = Join-Path "$BuildDir\python" $model.Name
        Copy-Item -Path $model.FullName -Destination $targetPath -Recurse -Force
        $copiedCount++
        Write-Host "✓ Copied model: $($model.Name)" -ForegroundColor Green
    }
    
    # Copy model utilities
    if (Test-Path "ml\utils") {
        Copy-Item -Path "ml\utils" -Destination "$BuildDir\python\utils" -Recurse -Force
        Write-Host "✓ Copied ML utilities" -ForegroundColor Green
    }
    
    Write-Host "✅ Copied $copiedCount model directories" -ForegroundColor Green
} else {
    Write-Host "⚠️  Models directory not found: $ModelsDir" -ForegroundColor Red
    exit 1
}

# Create model registry for Lambda layer
$registryContent = @'
"""Model registry for Lambda layer models"""
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LayerModelLoader:
    """Loads ML models from Lambda layer storage (zero S3 cost)"""
    
    def __init__(self):
        self.models_path = Path('/opt/python')
        self.loaded_models = {}
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan layer for available models"""
        self.available_models = {}
        if self.models_path.exists():
            for model_dir in self.models_path.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    # Look for ensemble_model.pkl in currency directory
                    model_file = model_dir / "ensemble_model.pkl"
                    if model_file.exists():
                        self.available_models[model_dir.name] = model_file
        
        logger.info(f"Lambda layer has {len(self.available_models)} pre-loaded models")
    
    def list_available_models(self):
        """List all models available in the layer"""
        return list(self.available_models.keys())
    
    def load_model(self, currency):
        """Load model from layer storage (instant access)"""
        if currency in self.loaded_models:
            return self.loaded_models[currency]
        
        if currency not in self.available_models:
            logger.warning(f"Model not found in layer: {currency}")
            return None
        
        try:
            model_path = self.available_models[currency]
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.loaded_models[currency] = model_data
            logger.info(f"Loaded layer model for {currency}")
            return model_data
        
        except Exception as e:
            logger.error(f"Failed to load layer model for {currency}: {e}")
            return None
    
    def get_model_info(self):
        """Get layer model information"""
        return {
            'total_models': len(self.available_models),
            'loaded_models': len(self.loaded_models),
            'available_currencies': self.list_available_models(),
            'layer_optimized': True,
            's3_cost': 0
        }

# Global layer model loader
_layer_loader = None

def get_layer_model_loader():
    """Get singleton layer model loader"""
    global _layer_loader
    if _layer_loader is None:
        _layer_loader = LayerModelLoader()
    return _layer_loader
'@

$registryContent | Out-File -FilePath "$BuildDir\python\model_registry.py" -Encoding UTF8

# Create ZIP file
Write-Host "🗜️  Creating layer ZIP file..." -ForegroundColor Yellow
$zipPath = Join-Path (Get-Location) $ZipFile

# Remove existing zip
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

# Create zip using .NET compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory("$BuildDir", $zipPath)

# Check size
$sizeBytes = (Get-Item $zipPath).Length
$sizeMB = [math]::Round($sizeBytes / 1MB, 2)

Write-Host "📏 Layer size: ${sizeMB}MB (limit: 250MB)" -ForegroundColor Cyan

if ($sizeMB -gt 250) {
    Write-Host "❌ Layer too large! Reduce model count or use EFS/Container approach" -ForegroundColor Red
    exit 1
}

# Publish layer to AWS
Write-Host "🚀 Publishing Lambda layer..." -ForegroundColor Yellow

$awsCommand = "aws lambda publish-layer-version --layer-name `"$LayerName`" --zip-file `"fileb://$zipPath`" --description `"PoEconomy ML Models - Static models for cost optimization`" --compatible-runtimes python3.11 python3.10"

if ($AwsProfile) {
    $awsCommand += " --profile $AwsProfile"
}

try {
    $result = Invoke-Expression $awsCommand | ConvertFrom-Json
    $layerArn = $result.LayerArn
    $layerVersion = $result.Version
    
    Write-Host "✅ Lambda layer created successfully!" -ForegroundColor Green
    Write-Host "📋 Layer ARN: $layerArn" -ForegroundColor Cyan
    Write-Host "🔢 Version: $layerVersion" -ForegroundColor Cyan
    
    # Save layer info for CloudFormation
    $layerInfo = @{
        LayerArn = $layerArn
        Version = $layerVersion
        SizeMB = $sizeMB
        ModelCount = $copiedCount
        CreatedAt = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    
    $layerInfo | ConvertTo-Json | Out-File -FilePath "layer-info.json" -Encoding UTF8
    Write-Host "💾 Layer info saved to layer-info.json" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Failed to publish layer: $_" -ForegroundColor Red
    Write-Host "💡 Make sure AWS CLI is configured and you have permissions" -ForegroundColor Yellow
    exit 1
}

# Cleanup
Remove-Item -Recurse -Force $BuildDir
Remove-Item -Force $zipPath

Write-Host ""
Write-Host "🎉 Done! Your models are now cached in Lambda layer for zero download costs." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update your Lambda function to use this layer ARN" -ForegroundColor White
Write-Host "2. Modify your Lambda code to use LayerModelLoader" -ForegroundColor White
Write-Host "3. Deploy updated CloudFormation template" -ForegroundColor White
Write-Host ""
Write-Host "Expected monthly savings: $50-200 in S3 transfer costs" -ForegroundColor Green 