# Use the official Lambda Python base image
FROM public.ecr.aws/lambda/python:3.11-x86_64

# Install system dependencies required for ML libraries (OpenMP support and build tools)
RUN yum update -y && \
    yum install -y libgomp gcc gcc-c++ make python3-devel && \
    yum clean all

# Install dependencies required for model inference
COPY aws/lambda/requirements.txt .
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt

# Copy Lambda handlers and shared ML modules
COPY aws/lambda /var/task/lambda
COPY ml /var/task/ml

# Copy models directly into the container (Lambda code storage)
COPY s3_backup/models /var/task/models

# Set environment variables
ENV PYTHONPATH=/var/task
ENV MODELS_DIR=/var/task/models

# Suppress XGBoost warnings and handle GPU issues
ENV XGBOOST_GPU_ID=-1
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONWARNINGS="ignore::UserWarning"

# Set working directory
WORKDIR /var/task

# Handler will be specified in CloudFormation ImageConfig.Command
