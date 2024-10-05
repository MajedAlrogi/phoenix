# Distributed BERT Training Setup

This project outlines the steps to set up a distributed environment for Neuronabox 2.0 / phoenix, using **PyTorch**, **NCCL**, and **CUDA**. This guide walks you through configuring the environment, building essential libraries, and verifying the setup with a toy test. More comprehensive evaluation instructions are in the eval folder.

---

## Table of Contents:
1. [Initialization](#Initialization)
2. 
   - [Configuration file (`config.sh`)](#configuration-file-configsh)
   - [Creating the environment (`create_env.sh`)](#creating-the-environment-create_envsh)
   - [Building NCCL](#building-nccl)
   - [Building PyTorch from Source](#building-pytorch-from-source)
   - [Final Test with Distributed BERT Training](#final-test-with-distributed-bert-training)

---

# Initialization

# ORI setup
### Configuration File (`config.sh`)

The `config.sh` script sets up the environment variables required for the project. This includes paths for the environment, CUDA, and NCCL, as well as setting up the distributed training configuration (master address and port). We will be creating a custom mamba/conda environment for this project. There will be two environments labelled ori and emu. One of which is an unmodified distributed environment using pytorch version v2.4.1 , NCCL version v2.10.3-1. The other environment wiill use a modified version  of pytorch and NCCL for emulation purposes.

Later, when building PyTorch and NCCL from source (using submodules), we will switch to these specific versions by checking out the appropriate tags in the submodules.



#### Pre-Build Steps:
1. make a directory in phoenix called ori
 ```bash
mkdir ori
```

2. export an environment variable called `$ENV_INSTALL_PATH` and `CUDA_HOME`
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
### building-nccl:
Simply invoke the script `scripts/build_nccl.sh`.
installation can be tested with `scripts/test_nccl.sh`
### building-pytorch-from-source:
Simply invoke the script `scripts/build_pytorch.sh`.
installation can be tested with `python scripts/test_nccl_ddp.py` (you migh need to install sympy: `mamba install sympy` )

### final-test-with-distributed-bert-training
make your way to eval/BERT and execute the command on two different nodes. This is to just verify everything works. Both nodes must activate the ori env and have the IP/port for master. Actual evaluation can be conducted through the docs in eval/BERT
 ```bash
cd eval/BERT
python train_bert_ddp.py --rank 0  # For the master node
python train_bert_ddp.py --rank 1  # For the worker node
```


