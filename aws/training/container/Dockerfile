FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install Python 3.11 and build tools
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make && \
    dnf clean all

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip

# Install dependencies required for model training
# Use same versions as Lambda container for consistency
COPY aws/training/container/requirements.txt .
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "pandas>=1.5.0,<2.0.0" \
    "python-dateutil>=2.8.0" \
    "pytz>=2023.3" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "lightgbm>=3.3.0" \
    "xgboost>=1.7.0" \
    "optuna>=2.10.0" \
    "joblib>=1.3.0" \
    "pyarrow>=10.0.0" \
    "boto3>=1.28.0" \
    "requests>=2.31.0"

# Copy the ML training code
COPY ml /app/ml

# Copy the training entrypoint
COPY aws/training/container/train_entrypoint.py /app/

# Set working directory
WORKDIR /app

# Set environment variables for proper logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Set the default command to run training
CMD ["python", "train_entrypoint.py"]