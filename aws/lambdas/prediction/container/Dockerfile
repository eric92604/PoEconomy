# Use the official Lambda Python base image (stable version)
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies required for ML libraries runtime
# Only runtime dependencies needed since we use binary wheels only
RUN yum update -y && \
    yum install -y libgomp && \
    yum clean all

# Install dependencies required for model inference
# CRITICAL: Force binary wheels only - GCC 7.3.1 is too old to compile modern packages
# Use exact versions to match training and feature engineering containers
COPY aws/lambdas/prediction/requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

# Copy Lambda handlers and shared ML modules
COPY aws/lambdas /var/task/lambda
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
