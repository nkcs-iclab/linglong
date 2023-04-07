# MCPT (PyTorch)

![version 0.5](https://img.shields.io/badge/version-0.5-blue)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch 2.0.0](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C?logo=pytorch&logoColor=white)

## Installation

1. Clone the repository.

    ```
    git clone https://github.com/alumik/mcpt-torch.git
    cd mcpt-torch
    ```

2. Install the package.

    ```
    pip install -e .
    ```

## Environment Setup

The required packages are not listed in `setup.py` yet, so you need to install them manually.

1. Create new conda environment with either `environment.yaml`.

    ```
    conda env create -f environment.yaml
    conda activate mcpt-torch
    pip install requirements.txt
    pip install requirements-torch.txt
    ```

2. Install Horovod (optional, for data parallel training).

    ```
    export NCCL_DIR=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nccl
    ln -s $NCCL_DIR/lib/libnccl.so.2 $NCCL_DIR/lib/libnccl.so
    HOROVOD_NCCL_INCLUDE=$NCCL_DIR/include/ \
      HOROVOD_NCCL_LIB=$NCCL_DIR/lib/ \
      HOROVOD_GPU_OPERATIONS=NCCL \
      HOROVOD_WITH_PYTORCH=1 \
      pip install --no-cache-dir horovod[pytorch]
    ```
    After successfully installing Horovod, run:

    ```
    horovodrun --check-build
    ```

    Every feature that was successfully enabled will be marked with an ‘X’. 
    If you intended to install Horovod with a feature that is not listed as enabled, you can reinstall Horovod, setting the appropriate environment variables to diagnose failures:

    ```
    pip uninstall horovod
    HOROVOD_WITH_...=1 pip install --no-cache-dir horovod
    ```
   
## Changelog

### 0.5 (prerelease)

- Stop batch text generation when the end of the text is reached in all samples.

### 0.4

- Added dataset scripts for fine-tuning.
- Introduced `mcpt.generate` function for generation.
- Introduced `mcpt.Model` class. This class can be used to load a specified model from a checkpoint.
- Replaced `[SEP]` with `\n` in generation results.
- Exported `mcpt.Tokenizer` and `mcpt.PinyinTokenizer` to the top-level module.
- Fixed the training script by adding if statements to prevent missing object/attribute/reference errors when using mixed precision training or data parallel training.
- Fixed the model saving callback.

### 0.3

- Added modules for evaluation.
- Refactored `mcpt/sampling.py`. The `Sampler` class has now been moved to `mcpt/generation.py`.

## Copyright

© 2023 College of Software, Nankai University All Rights Reserved
