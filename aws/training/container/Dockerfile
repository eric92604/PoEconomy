FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install Python 3.11 and build tools
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make && \
    dnf clean all

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip

# Install dependencies required for model training
# Use exact versions to match prediction and feature engineering containers
COPY aws/training/container/requirements.txt .

# Upgrade pip and install wheel first to avoid build issues
RUN pip install --upgrade pip setuptools wheel

# Install packages with retry logic and better error handling
RUN pip install --no-cache-dir --retries 3 --timeout 300 \
    --upgrade-strategy only-if-needed \
    -r requirements.txt

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