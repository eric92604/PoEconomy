#!/bin/bash
"""
Create Lambda Layer with ML Models for Cost Optimization

This script packages trained ML models into Lambda layers, which:
- Provides persistent storage across invocations (no S3 download costs)
- Allows models up to 250MB total
- Shared across multiple Lambda functions
- Versioned for easy rollbacks
"""

set -e

# Configuration
LAYER_NAME="poeconomy-ml-models"
MODELS_DIR="ml/models/currency_production"
BUILD_DIR="layer-build"
ZIP_FILE="ml-models-layer.zip"

echo "🚀 Building Lambda Layer with ML Models..."

# Clean and create build directory
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR/python

# Copy essential models (prioritize by size/usage)
echo "📦 Copying model files..."
if [ -d "$MODELS_DIR" ]; then
    # Copy only the most important models to stay under 250MB
    find "$MODELS_DIR" -name "*.pkl" -o -name "*.joblib" -o -name "*.json" | \
    head -50 | \
    xargs -I {} cp {} $BUILD_DIR/python/
    
    # Copy model metadata
    cp -r ml/utils/ $BUILD_DIR/python/ 2>/dev/null || true
    
    # Create model registry
    cat > $BUILD_DIR/python/model_registry.py << 'EOF'
"""Model registry for Lambda layer models"""
import os
import pickle
import json
from pathlib import Path

class LayerModelLoader:
    def __init__(self):
        self.models_path = Path('/opt/python')
        self.loaded_models = {}
    
    def list_available_models(self):
        """List all models available in the layer"""
        return list(self.models_path.glob('*.pkl')) + list(self.models_path.glob('*.joblib'))
    
    def load_model(self, currency):
        """Load model from layer storage"""
        if currency in self.loaded_models:
            return self.loaded_models[currency]
        
        model_files = [
            f"{currency}_model.pkl",
            f"{currency.lower()}_model.pkl",
            f"{currency.replace(' ', '_')}_model.pkl"
        ]
        
        for model_file in model_files:
            model_path = self.models_path / model_file
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.loaded_models[currency] = model
                    return model
                except Exception as e:
                    print(f"Failed to load {model_file}: {e}")
        
        return None
EOF
    
    echo "✅ Copied $(find $BUILD_DIR/python -name "*.pkl" -o -name "*.joblib" | wc -l) model files"
else
    echo "⚠️  Models directory not found: $MODELS_DIR"
    exit 1
fi

# Create ZIP file
echo "🗜️  Creating layer ZIP file..."
cd $BUILD_DIR
zip -r "../$ZIP_FILE" . -q
cd ..

# Check size
SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
SIZE_MB=$((SIZE / 1024 / 1024))

echo "📏 Layer size: ${SIZE_MB}MB (limit: 250MB)"

if [ $SIZE_MB -gt 250 ]; then
    echo "❌ Layer too large! Reduce model count or use EFS/Container approach"
    exit 1
fi

# Publish layer
echo "🚀 Publishing Lambda layer..."
aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --zip-file "fileb://$ZIP_FILE" \
    --description "PoEconomy ML Models - Static models for cost optimization" \
    --compatible-runtimes python3.11 python3.10

echo "✅ Lambda layer created successfully!"
echo "💡 Add this layer ARN to your Lambda functions in CloudFormation"

# Cleanup
rm -rf $BUILD_DIR $ZIP_FILE

echo "🎉 Done! Your models are now cached in Lambda layer for zero download costs." 