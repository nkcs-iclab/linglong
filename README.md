# LingLong (玲珑): A Small-Scale Chinese PreTrained Language Model

![version 0.10.3](https://img.shields.io/badge/version-0.10.3-blue)
![Python >=3.11,<3.12](https://img.shields.io/badge/Python->=3.11,<3.12-blue?logo=python&logoColor=white)
![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![License GNU GPL v3](https://img.shields.io/badge/License-GNU%20GPL%20v3-blue?logo=gnu&logoColor=white)

This is LingLong (玲珑), a Chinese pretrained language model trained by the College of Software at Nankai University.
Built on the foundation of the GPT-3 architecture, it has been meticulously trained on a vast collection of private
datasets.
With a modest parameter count of only 317 million, LingLong is significantly smaller than the original GPT-3 model, yet
it delivers impressive results across various NLP tasks.
In Chinese, "玲珑" translates to "exquisite" or "delicate," which perfectly embodies the small yet mighty nature of this
model.
Therefore, we chose to name it "LingLong" in honor of its exceptional precision and finesse.

Although it's true that this model's performance isn't on par with the large pretrained language models boasting
hundreds of billions of parameters, its relatively low parameter count makes it accessible to researchers with limited
computing resources.
As a result, this model serves as an excellent foundation for conducting follow-up research, such as fine-tuning.
By utilizing this model, researchers can begin to delve into the intricacies of pretrained language models and to
unravel the mysteries of language processing without the need for excessive computational resources.

## Hardware Requirements

The following hardware is recommended for training:

- NVIDIA Tesla V100 32GB GPUs (or any other GPUs with at least 24 GB of memory)

The following hardware is recommended for inference:

- NVIDIA Tesla T4 16GB GPUs (or any other GPUs with at least 4 GB of memory)

The model can also run on CPUs, but the training and inference speed will be significantly slower.

## Python Requirements

This package requires Python 3.11 or later.

## Environment Setup

The required packages are not listed in `setup.py` yet, so you need to install them manually.

1. Clone the repository.

    ```
    git clone https://github.com/NKCSICLab/linglong.git
    cd linglong
    ```

2. Create new conda environment with `environment.yaml`.

    ```
    conda env create -f environment.yaml
    conda activate linglong
    ```

3. Install the required packages. Be sure to install PyTorch first. You have to edit the `requirements-torch.txt`
   and `requirements.txt` file to match your CUDA version. The default version is 12.1.

    ```
    pip install -r requirements-torch.txt
    pip install -r requirements.txt
    ```

   *Optional:* If you want to perform evaluation on public datasets, you need to install the evaluation dependencies.

    ```
    pip install -r requirements-evaluation.txt
    ```

4. Install the package.

    ```
    pip install -e .
    ```

5. Install DeepSpeed (optional, for DeepSpeed enabled training).

    ```
    pip install ninja deepspeed
    ```

   After installation, you can validate your installation and see which ops your machine is compatible with via the
   DeepSpeed environment report with `ds_report` or `python -m deepspeed.env_report`.

## A Quick Guide to Text Generation

We provide an interactive text generation script `generation/generate.py` for generating text from a trained model.

```
python generate.py --model=/path/to/linglong/model
```

There is also a script `generation/api-example.py` demonstrating how to use the generation API.

More usage details can be found using the `--help` flag.

## Pretrained Models

### Legacy Models

You have to convert these legacy models to the latest format before using them with the current version of the codebase.
A conversion script is provided at `utils/torch2transformers.sh`.

| Model Name          | Version | Parameters | Size   | Download                                                                |
|---------------------|---------|------------|--------|-------------------------------------------------------------------------|
| LingLong            | V12     | 317 M      | 1.2 GB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUj-wymU62HcCOduEZcg?e=bzyCzU) |
| LingLong-Backward   | V1      | 317M       | 1.2 GB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUkqtni8g7Xkr_JuGuqg?e=AMLWUh) |
| LingLong-Pinyin     | V1      | 318M       | 1.2 GB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUkqto9guYQ0BZLVTyzw?e=eKh7H4) |
| LingLong-Small      | V1      | 106M       | 367 MB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUkqtlbLLOx0t03obH1w?e=ikRx63) |
| LingLong-Small-Word | V1      | 106M       | 404 MB | [OneDrive](https://1drv.ms/u/s!AszCaIeLPgHUkqtmk8xMs-OmBwhtdw?e=mlXZGf) |

### Latest Models

| Model Name    | Version | Parameters | Size    | Download                                    |
|---------------|---------|------------|---------|---------------------------------------------|
| LingLong-317M | V12     | 317 M      | 1.27 GB | https://huggingface.co/AlumiK/LingLong-317M |

## Changelog

### 0.11 (Upcoming)

- *Dataset:* Dataset pre-processing scripts can now initialize tokenizers from pretrained models or vocab files.
- *Dataset:* Add a streaming pre-training dataset class `StreamingPretrainingDataset`.
- *Evaluation:* Add a new evaluation module.
- *Generation:* Add `bingsearch` plugin for text generation.
- Rename `LingLongLMHeadModel` to `LingLongForCausalLM`.
- Add BOS and EOS tokens to tokenizers.
- Add a new model conversion script.
- Progress bars are now printed to `stderr` instead of `stdout`.
- Use transformers' logger instead of the built-in `warnings` module.
- Add example scripts for all modules.

### 0.10

- Hello 🤗 Transformers! We have migrated to the Hugging Face Transformers library.
- Remove the `mcpt` package and replace it with the `linglong` package.
- Remove RLHF support. This feature will be re-implemented in the future.
- Remove all experimental features. These features will be considered for re-implementation in the future.
- *Evaluation:* Remove the evaluation module. This module will be re-implemented in the future.

### 0.9

- *Training:* Allow users to skip steps during training.
- *Training:* Add `save_initial` and `save_final` switches to the training script.
- *Evaluation:* Add NER datasets and metrics.
- Various bug fixes for the latest dependencies.
- Migrate from `setup.py` to `pyproject.toml`.

### 0.8

- *Dataset:* Add processing scripts for plain text pre-training data.
- *Training:* Fix a bug that caused the training not able to find meta files.
- *Training:* Allow users to disable the strict mode when loading the model.
- *Training:* It is now possible to add a prefix to the name of the saved model.
- Update the format of the model output from `tuple` to `dict`.
- Add RLHF (stage 1 & stage 2) support.
- Move the LM head from the basic model to the model wrapper `mcpt.Model`. You can now retrieve the hidden states from
  the model wrapper using `mcpt.Model.hidden_states`.

### 0.7

- *Evaluation:* Add more evaluation dataset and metrics.
- *Evaluation:* Update the evaluation config schema.
- *Evaluation:* Various bug fixes.
- *Experimental:* Add a word-based tokenizer and a word-based vocabulary file (from CPM-2).
- Rename `mcpt.print_dict` to `mcpt.pprint`.
- Compress tfrecord files with gzip to save disk space.
- Convert meta files from pickle to json.

### 0.6

- *Dataset:* Remove the template list from dataset classes.
- *Dataset:* The templates in the dataset classes now accept a list of strings.
- *Training:* *DeepSpeed:* Fix model saving issue with DeepSpeed models.
- *Generation:* Add prompt plugin support for text generation.
- *Experimental:* Add more experimental dataset classes.
- The code is now compatible with Python 3.6.
- Various stability improvements.

### 0.5

- *Training:* *DeepSpeed:* Add `train-ds.py` for DeepSpeed enabled training.
- *Generation:* Stop batch text generation when the end of the text is reached in all samples.
- *Generation:* Fixed: Text are now clipped to the maximum context length of the model.
- Move `use_pinyin` and `backward` arguments from method arguments to the model configuration.

### 0.4

- *Dataset:* Add dataset scripts for fine-tuning.
- *Training:* Fix the training script by adding if statements to prevent missing object/attribute/reference errors
  when using mixed precision training or data parallel training.
- *Training:* Fix the model saving callback.
- *Generation:* Introduce `mcpt.generate` function for generation.
- *Generation:* Replace `[SEP]` with `\n` in generation results.
- Introduce `mcpt.Model` class. This class can be used to load a specified model from a checkpoint.
- Export `mcpt.Tokenizer` and `mcpt.PinyinTokenizer` to the top-level module.

### 0.3

- *Dataset:* Add modules for evaluation.
- *Generation:* Refactor `mcpt/sampling.py`. The `Sampler` class has now been moved to `mcpt/generation.py`.

## Copyright

© 2023-2024 College of Software, Nankai University All Rights Reserved
