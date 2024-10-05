# Distributed BERT Training Setup

This project outlines the steps to set up a distributed environment for fine-tuning BERT on multiple GPUs, using **PyTorch**, **NCCL**, and **CUDA**. This guide walks you through configuring the environment, building essential libraries, and verifying the setup with a toy test.

---

## Table of Contents:
1. [Project Setup](#project-setup)
   - [Configuration file (`config.sh`)](#configuration-file-configsh)
   - [Creating the environment (`create_env.sh`)](#creating-the-environment-create_envsh)
2. [Building NCCL](#building-nccl)
   - [Cloning and building NCCL](#cloning-and-building-nccl)
   - [Testing NCCL](#testing-nccl)
3. [Building PyTorch from Source](#building-pytorch-from-source)
4. [Final Test with Distributed BERT Training](#final-test-with-distributed-bert-training)

---

## Project Setup

### Configuration File (`config.sh`)

The `config.sh` script sets up the environment variables required for the project. This includes paths for the environment, CUDA, and NCCL, as well as setting up the distributed training configuration (master address and port).

#### Steps:
1. Create a `config.sh` file with the following contents:

```bash
#!/bin/bash

# Path to your environment installation
export ENV_INSTALL_PATH=/data/fat/neuronabox/phoenix/ori

# Path to your CUDA installation (inside the custom environment)
export CUDA_HOME=/data/fat/neuronabox/phoenix/ori

# Add environment binaries to PATH
export PATH=$ENV_INSTALL_PATH/bin:$PATH

# Add environment libraries to the library search path
export LD_LIBRARY_PATH=$ENV_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Add CUDA binaries to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA libraries to the library search path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Distributed training configuration
export MASTER_ADDR="172.18.0.47"  # Replace with your master node's IP address
export MASTER_PORT="12345"         # Choose an arbitrary available port
export WORLD_SIZE=2

echo "Environment variables set for distributed training."
eval "$(conda shell.bash hook)"
conda activate $ENV_INSTALL_PATH
```
