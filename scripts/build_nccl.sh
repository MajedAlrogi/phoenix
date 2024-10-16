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
    git clone https://github.com/MajedAlrogi/nccl-phoenix-2.0 nccl
fi

# Navigate into the NCCL directory
cd nccl

# Check out the version of NCCL you want to build (BASED ON v2.19.4-1)
if [[ "$ENV_INSTALL_PATH" == *"ori"* ]]; then
    NCCL_VERSION="ori"
elif [[ "$ENV_INSTALL_PATH" == *"emu"* ]]; then
    NCCL_VERSION="emu"
else
    echo "No valid branch found for the given ENV_INSTALL_PATH. Please make sure it contains 'ori' or 'emu'."
    exit 1
fi

git checkout $NCCL_VERSION

# Build NCCL with CUDA support
make -j$(nproc) src.build CUDA_HOME=$CUDA_HOME

# Install NCCL into the custom environment path
make install PREFIX=$ENV_INSTALL_PATH

echo "NCCL version $NCCL_VERSION has been built and installed in $ENV_INSTALL_PATH."

