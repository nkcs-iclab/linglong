# Mini-Scale Chinese PreTrained Language Model (MCPT)

![version 0.6](https://img.shields.io/badge/version-0.6-blue)
![Python >=3.6,<3.12](https://img.shields.io/badge/Python->=3.6,<3.12-blue?logo=python&logoColor=white)
![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?logo=tensorflow&logoColor=white)

## Python Requirements

This package requires Python 3.6 or later, with a few exceptions:

- If you want to use the parallel evaluation script, you need Python 3.11 or later.
- PyTorch 2.0 requires Python 3.8 or later. PyTorch with a lower version number may work, but it is not tested.
- TensorFlow 2.12 requires Python 3.8 or later. TensorFlow with a lower version number may work, but it is not tested.

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
    export NCCL_HOME=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/nvidia/nccl
    ln -s $NCCL_HOME/lib/libnccl.so.2 $NCCL_HOME/lib/libnccl.so
    HOROVOD_NCCL_HOME=$NCCL_HOME \
      HOROVOD_GPU_OPERATIONS=NCCL \
      HOROVOD_WITH_PYTORCH=1 \
      pip install --no-cache-dir horovod[pytorch]
    ```
    After successfully installing Horovod, run:

    ```
    horovodrun --check-build
    ```

    Every feature that was successfully enabled will be marked with an "X".
   
3. Install DeepSpeed (optional, experimental, for DeepSpeed enabled training).

    ```
    pip install deepspeed
    ```
   
    After installation, you can validate your installation and see which ops your machine is compatible with via the DeepSpeed environment report with `ds_report` or `python -m deepspeed.env_report`.

## Changelog

### 0.6 (pre-release)

- *Compatibility* The code is now compatible with Python 3.6.
- *Dataset:* Removed the template list from dataset classes.
- *Dataset:* The templates in the dataset classes now accept a list of strings.
- *Training:* *DeepSpeed:* Fixed model saving issue with DeepSpeed models

### 0.5

- *Training:* *DeepSpeed:* Added `train-ds.py` for DeepSpeed enabled training.
- *Generation:* Stop batch text generation when the end of the text is reached in all samples.
- Moved `use_pinyin` and `backward` arguments from method arguments to the model configuration.
- *Generation:* Fixed: Text are now clipped to the maximum context length of the model.

### 0.4

- *Dataset:* Added dataset scripts for fine-tuning.
- *Generation:* Introduced `mcpt.generate` function for generation.
- Introduced `mcpt.Model` class. This class can be used to load a specified model from a checkpoint.
- *Generation:* Replaced `[SEP]` with `\n` in generation results.
- Exported `mcpt.Tokenizer` and `mcpt.PinyinTokenizer` to the top-level module.
- *Training:* Fixed the training script by adding if statements to prevent missing object/attribute/reference errors when using mixed precision training or data parallel training.
- *Training:* Fixed the model saving callback.

### 0.3

- *Dataset:* Added modules for evaluation.
- *Generation:* Refactored `mcpt/sampling.py`. The `Sampler` class has now been moved to `mcpt/generation.py`.

## Copyright

© 2023 College of Software, Nankai University All Rights Reserved
