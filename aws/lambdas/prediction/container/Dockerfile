# Use the official Lambda Python base image (stable version)
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies required for ML libraries runtime
# Only runtime dependencies needed since we use binary wheels only
RUN yum update -y && \
    yum install -y libgomp && \
    yum clean all

# Install dependencies required for model inference
# CRITICAL: Force binary wheels only - GCC 7.3.1 is too old to compile modern packages
COPY aws/lambda/requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: \
    "numpy>=1.21.0,<2.0.0" \
    "pandas>=1.5.0,<2.0.0" \
    "python-dateutil>=2.8.0" \
    "pytz>=2023.3" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "xgboost>=1.7.0,<=2.1.0" \
    "optuna>=2.10.0" \
    "joblib>=1.3.0" \
    "pyarrow>=10.0.0" \
    "boto3>=1.28.0" \
    "requests>=2.31.0" \
    "lightgbm>=3.3.0,<4.0.0"

# Copy Lambda handlers and shared ML modules
COPY aws/lambda /var/task/lambda
COPY ml /var/task/ml

# Copy models directly into the container (Lambda code storage)
COPY s3_backup/models /var/task/models

# Set environment variables for proper logging and ML inference
ENV PYTHONPATH=/var/task
ENV MODELS_DIR=/var/task/models
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Suppress XGBoost warnings and handle GPU issues
ENV XGBOOST_GPU_ID=-1
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONWARNINGS="ignore::UserWarning"

# Set working directory
WORKDIR /var/task

# Handler will be specified in CloudFormation ImageConfig.Command
