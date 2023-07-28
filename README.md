# LingLong (玲珑): A Small-Scale Chinese PreTrained Language Model

![version 0.7.0](https://img.shields.io/badge/version-0.7.0-blue)
![Python >=3.6,<3.12](https://img.shields.io/badge/Python->=3.6,<3.12-blue?logo=python&logoColor=white)
![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?logo=tensorflow&logoColor=white)
![License GNU GPL v3](https://img.shields.io/badge/License-GNU%20GPL%20v3-blue?logo=gnu&logoColor=white)

This is LingLong (玲珑), a Chinese pretrained language model trained by the College of Software at Nankai University.
Built on the foundation of the GPT-3 architecture, it has been meticulously trained on a vast collection of private datasets.
With a modest parameter count of only 317 million, LingLong is significantly smaller than the original GPT-3 model, yet it delivers impressive results across various NLP tasks.
In Chinese, "玲珑" translates to "exquisite" or "delicate," which perfectly embodies the small yet mighty nature of this model.
Therefore, we chose to name it "LingLong" in honor of its exceptional precision and finesse.

Although it's true that this model's performance isn't on par with the large pretrained language models boasting hundreds of billions of parameters, its relatively low parameter count makes it accessible to researchers with limited computing resources.
As a result, this model serves as an excellent foundation for conducting follow-up research, such as fine-tuning.
By utilizing this model, researchers can begin to delve into the intricacies of pretrained language models and to unravel the mysteries of language processing without the need for excessive computational resources.

## Hardware Requirements

The following hardware is recommended for training:

- NVIDIA Tesla V100 32GB GPUs (or any other GPUs with at least 24 GB of memory)

The following hardware is recommended for inference:

- NVIDIA Tesla T4 16GB GPUs (or any other GPUs with at least 4 GB of memory)

The model can also run on CPUs, but the training and inference speed will be significantly slower.

## Python Requirements

This package requires Python 3.6 or later, with a few exceptions:

- If you want to use the parallel evaluation script, you need Python 3.11 or later.
- PyTorch 2.0 requires Python 3.8 or later. PyTorch with a lower version number may work, but it is not tested.
- TensorFlow 2.12 requires Python 3.8 or later. TensorFlow with a lower version number may work, but it is not tested.

## Environment Setup

The required packages are not listed in `setup.py` yet, so you need to install them manually.

1. Clone the repository.

    ```
    git clone https://github.com/NKCSICLab/linglong-mcpt.git
    cd linglong-mcpt
    ```

2. Create new conda environment with `environment.yaml`.

    ```
    conda env create -f environment.yaml
    conda activate mcpt-torch
    pip install -r requirements.txt
    pip install -r requirements-torch.txt
    ```
   
    *Optional:* If you want to perform evaluation on public datasets, you need to install the evaluation dependencies.
   
    ```
    pip install -r requirements-evaluation.txt
    ```

3. Install the package.

    ```
    pip install -e .
    ```

4. Install Horovod (optional, for data parallel training).

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
   
5. Install DeepSpeed (optional, experimental, for DeepSpeed enabled training).

    ```
    pip install deepspeed
    ```
   
    After installation, you can validate your installation and see which ops your machine is compatible with via the DeepSpeed environment report with `ds_report` or `python -m deepspeed.env_report`.

## A Quick Guide to Text Generation

We provide an interactive text generation script `generation/generate.py` for generating text from a trained model.

```
python generate.py \
  --model=/path/to/linglong-mcpt/models/V12.pt \
  --model-config=/path/to/linglong-mcpt/common/model-configs/317M-WSZ1024L24.yaml
```

There is also a script `generation/api-example.py` demonstrating how to use the generation API.

## Pretrained Models

| Model Name    | Version | Parameters | Size   | Download                                                                |
|---------------|---------|------------|--------|-------------------------------------------------------------------------|
| LingLong      | V12     | 317 M      | 1.2 GB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUj-wymU62HcCOduEZcg?e=bzyCzU) |
| LingLong-Chat | V4      | 317 M      | 1.2 GB | TBD                                                                     |     

## Changelog

### 0.8 (pre-release)

- Updated the format of the model output from `tuple` to `dict`.

### 0.7

- *Experimental:* Added a word-based tokenizer and a word-based vocabulary file (from CPM-2).
- *Evaluation:* Added more evaluation dataset and metrics.
- *Evaluation:* Updated the evaluation config schema.
- *Evaluation:* Various bug fixes.
- Renamed `mcpt.print_dict` to `mcpt.pprint`.
- Compressed tfrecord files with gzip to save disk space.
- Converted meta files from pickle to json.

### 0.6

- *Compatibility:* The code is now compatible with Python 3.6.
- *Stability:* Various stability improvements.
- *Dataset:* Removed the template list from dataset classes.
- *Dataset:* The templates in the dataset classes now accept a list of strings.
- *Training:* *DeepSpeed:* Fixed model saving issue with DeepSpeed models.
- *Generation:* Added prompt plugin support for text generation.
- *Experimental:* Added more experimental dataset classes.

### 0.5

- *Training:* *DeepSpeed:* Added `train-ds.py` for DeepSpeed enabled training.
- *Generation:* Stop batch text generation when the end of the text is reached in all samples.
- *Generation:* Fixed: Text are now clipped to the maximum context length of the model.
- Moved `use_pinyin` and `backward` arguments from method arguments to the model configuration.

### 0.4

- *Dataset:* Added dataset scripts for fine-tuning.
- *Training:* Fixed the training script by adding if statements to prevent missing object/attribute/reference errors when using mixed precision training or data parallel training.
- *Training:* Fixed the model saving callback.
- *Generation:* Introduced `mcpt.generate` function for generation.
- *Generation:* Replaced `[SEP]` with `\n` in generation results.
- Introduced `mcpt.Model` class. This class can be used to load a specified model from a checkpoint.
- Exported `mcpt.Tokenizer` and `mcpt.PinyinTokenizer` to the top-level module.

### 0.3

- *Dataset:* Added modules for evaluation.
- *Generation:* Refactored `mcpt/sampling.py`. The `Sampler` class has now been moved to `mcpt/generation.py`.

## Copyright

© 2023 College of Software, Nankai University All Rights Reserved
