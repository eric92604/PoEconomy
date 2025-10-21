# Use the official Lambda Python base image (stable version)
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies required for ML libraries runtime
RUN yum update -y && \
    yum install -y libgomp && \
    yum clean all

# Install dependencies required for ingestion operations
# CRITICAL: Force binary wheels only - GCC 7.3.1 is too old to compile modern packages
COPY aws/lambdas/ingestion/container/requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

# Copy shared ML modules
COPY ml /var/task/ml

# Copy Lambda handlers and shared ML modules
COPY aws/lambdas /var/task/lambda

# Set environment variables for proper logging and ML operations
ENV PYTHONPATH=/var/task
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Suppress warnings
ENV PYTHONWARNINGS="ignore::UserWarning"

# Set working directory
WORKDIR /var/task

# Handler will be specified in CloudFormation ImageConfig.Command
