# Distributed BERT Training Setup

This project outlines the steps to set up a distributed environment for Neuronabox 2.0 / phoenix, using **PyTorch**, **NCCL**, and **CUDA**. This guide walks you through configuring the environment, building essential libraries, and verifying the setup with a toy test. More comprehensive evaluation instructions are in the eval folder.

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

The `config.sh` script sets up the environment variables required for the project. This includes paths for the environment, CUDA, and NCCL, as well as setting up the distributed training configuration (master address and port). We will be creating a custom mamba/conda environment for this project. There will be two environments labelled ori and emu. One of which is an unmodified distributed environment using pytorch version v2.4.1 , NCCL version v2.10.3-1. The other environment wiill use a modified version  of pytorch and NCCL for emulation purposes.

Later, when building PyTorch and NCCL from source (using submodules), we will switch to these specific versions by checking out the appropriate tags in the submodules.

# ORI setup

#### Steps:
1. make a directory in phoenix called ori
 ```bash
mkdir ori
```

2. export an environment variable called $ENV_INSTALL_PATH and CUDA_HOME
```bash
# Path to your environment installation
export ENV_INSTALL_PATH=[path to phoenix]/ori

# Path to your CUDA installation (inside the custom environment)
export CUDA_HOME=[path to phoenix]/ori
```
3. call the script scripts/create_env.sh to create the environment
 ```bash
 bash ./scripts/create_env.sh $ENV_PATH # takes about 30 minutes
```

4. Create a `config.sh` file with the following contents:

```bash
#!/bin/bash

# Path to your environment installation
export ENV_INSTALL_PATH=[path to phoenix]/ori

# Path to your CUDA installation (inside the custom environment)
export CUDA_HOME=[path to phoenix]/ori

# Add environment binaries to PATH
export PATH=$ENV_INSTALL_PATH/bin:$PATH

# Add environment libraries to the library search path
export LD_LIBRARY_PATH=$ENV_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Add CUDA binaries to PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add CUDA libraries to library search path
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=[path to phoenix]/cache/transformers
echo "Environment variables have been set."

# Modify the following to use the master node's IP address and an available port
export MASTER_ADDR="[FILL]"  # Replace with the IP address of the master node
export MASTER_PORT="[FILL]"         # Choose an arbitrary available port
export WORLD_SIZE=[FILL]
export OMPI_COMM_WORLD_SIZE=[FILL]
echo "cluster config has been set"

eval "$(conda shell.bash hook)"
conda activate $ENV_INSTALL_PATH

echo "activated environment!"
```

5. invoke the config
 ```bash
source config.sh
```
6. 
