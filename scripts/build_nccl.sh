#!/bin/bash

# Check if ENV_INSTALL_PATH is set
if [ -z "$ENV_INSTALL_PATH" ]; then
    echo "ENV_INSTALL_PATH is not set. Please source config.sh first."
    exit 1
fi

# Check if CUDA_HOME is set
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set. Please source config.sh first."
    exit 1
fi

# Enable exit on error and command echo for easier debugging
set -ex

# Clone NCCL repository if not already present
if [ ! -d "nccl" ]; then
    git clone https://github.com/NVIDIA/nccl.git
fi

# Navigate into the NCCL directory
cd nccl

# Check out the version of NCCL you want to build (replace with the desired version)
NCCL_VERSION="v2.10.3-1"
git checkout $NCCL_VERSION

# Build NCCL with CUDA support
make -j src.build CUDA_HOME=$CUDA_HOME

# Install NCCL into the custom environment path
make install PREFIX=$ENV_INSTALL_PATH

echo "NCCL version $NCCL_VERSION has been built and installed in $ENV_INSTALL_PATH."

