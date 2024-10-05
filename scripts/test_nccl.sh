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

# Clone NCCL tests if not already present
if [ ! -d "nccl-tests" ]; then
    git clone https://github.com/NVIDIA/nccl-tests.git
fi

# Navigate into the NCCL tests directory
cd nccl-tests

# Build the NCCL tests using CUDA
make CUDA_HOME=$CUDA_HOME

# Determine how many GPUs are available
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ $NUM_GPUS -eq 0 ]; then
    echo "No GPUs available for NCCL testing."
    exit 1
elif [ $NUM_GPUS -eq 1 ]; then
    echo "Only 1 GPU detected, running single GPU test."
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
else
    echo "$NUM_GPUS GPUs detected, running multi-GPU test."
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g $NUM_GPUS
fi

