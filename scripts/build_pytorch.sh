#!/bin/bash

# Ensure the ENV_INSTALL_PATH is set from config.sh
if [ -z "$ENV_INSTALL_PATH" ]; then
    echo "ENV_INSTALL_PATH is not set. Please source config.sh first."
    exit 1
fi

# Activate the environment
eval "$(conda shell.bash hook)" 
conda activate $ENV_INSTALL_PATH

set -ex

# Step 1: Install additional dependencies needed to build PyTorch
echo "Installing additional dependencies for PyTorch..."
mamba install -c conda-forge libuv protobuf openmpi cmake ninja pyyaml typing_extensions

# Step 2: Clone the PyTorch repository if it doesn't exist
echo "Cloning the PyTorch repository..."
if [ ! -d "pytorch" ]; then
    git clone --recursive https://github.com/pytorch/pytorch.git
fi
cd pytorch

# Step 3: Checkout the specific version of PyTorch (2.4.1)
PYTORCH_VERSION="v2.4.1"  # Set to PyTorch version 2.4.1
echo "Checking out PyTorch version: $PYTORCH_VERSION"
git fetch --all --tags
git checkout $PYTORCH_VERSION
git submodule sync
git submodule update --init --recursive

# Step 4: Set up environment variables for building PyTorch
echo "Setting up environment variables..."
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export NCCL_ROOT=$ENV_INSTALL_PATH
export PATH=$ENV_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ENV_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Step 5: Clean previous builds if any
echo "Cleaning previous PyTorch builds..."
python setup.py clean

# Step 6: Build and install PyTorch
echo "Building and installing PyTorch with NCCL and CUDA support..."
python setup.py install

echo "PyTorch build completed!"

