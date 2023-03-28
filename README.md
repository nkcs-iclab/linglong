# MCPT (PyTorch)

![version 0.2](https://img.shields.io/badge/version-0.2-blue)
![Python 3.10-3.11](https://img.shields.io/badge/Python->=3.10,<=3.11-blue?logo=python&logoColor=white)
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
    ```

2. Install Horovod (optional, for data parallel training).

    ```
    HOROVOD_NCCL_INCLUDE=$CONDA_PREFIX/include/ \
      HOROVOD_NCCL_LIB=$CONDA_PREFIX/lib/ \
      HOROVOD_GPU_OPERATIONS=NCCL \
      HOROVOD_WITH_PYTORCH=1 \
      pip install horovod[pytorch]
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

## Copyright

© 2023 College of Software, Nankai University All Rights Reserved