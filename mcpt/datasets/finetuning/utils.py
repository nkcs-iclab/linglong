import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    datasets = {
        'Math23K': mcpt.datasets.finetuning.datasets.Math23KDataset,
    }
    experimental_datasets = {
        'CustomQA': mcpt.datasets.finetuning.datasets.CustomQADataset,
    }
    datasets = mcpt.merge_configs(datasets, experimental_datasets)
    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    return datasets[config['dataset']](
        input_path=input_path,
        output_path=output_path,
        vocab_path=config['vocab'],
        pinyin_vocab_path=config['pinyin_vocab'],
        template_id=config['template_id'],
        model_config=config['model_config'],
        special_tokens=config['special_tokens'],
        split=config['split'],
        use_cache=config['use_cache'],
        items_per_file=config['items_per_file'],
        extra_config=config.get('extra_config', None),
    )
