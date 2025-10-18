FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install Python 3.11 and build tools
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-pip python3.11-devel gcc gcc-c++ make && \
    dnf clean all

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip

# Install dependencies required for feature engineering
# Use same versions as Lambda/Training containers for consistency
COPY aws/feature-engineering/container/requirements.txt .
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "pandas>=1.5.0,<2.0.0" \
    "python-dateutil>=2.8.0" \
    "pytz>=2023.3" \
    "scikit-learn>=1.3.0" \
    "boto3>=1.28.0" \
    "requests>=2.31.0" \
    "pyarrow>=10.0.0"

# Copy the ML training code
COPY ml /var/task/ml

# Copy the feature engineering entrypoint
COPY aws/feature-engineering/container/feature_engineering_entrypoint.py /var/task/

# No test scripts needed for production

# Set working directory
WORKDIR /var/task

# Set the default command to run feature engineering
CMD ["python", "feature_engineering_entrypoint.py"]
