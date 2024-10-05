# BERT                                                                                                                                      
We fine-tune a pretrained BERT model on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. This script is based on [this repo](https://github.com/sands-lab/omnireduce-experiments/tree/master/models/BERT). The checkpoint needs to be placed in `./dataset/checkpoint`. Instructions are below:

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `tqdm`, `dllogger` and `apex`.

**Install Dependencies** :
```bash
    pip3 install --upgrade pip
    pip3 install packaging
    pip install six
    pip install tqdm
    pip install nvidia-pyindex
    pip install nvidia-dllogger

    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

**Dowload model checkpoint** :
```bash
    cd ./dataset/checkpoint
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_qa_squad11_amp/versions/19.09.0/zip -O bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    unzip bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    cd ../../ && mkdir -p results
```

## BERT Training

###  Run workers
Worker 0:
```bash
. ../../config.sh
OMPI_COMM_WORLD_RANK=0 MOD_KERNEL_BYPASS=0 ./run.sh
```
Worker 1:
```bash
. ../../config.sh
OMPI_COMM_WORLD_RANK=1 MOD_KERNEL_BYPASS=0 ./run.sh
```
