import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    datasets = {
        'Math23K': mcpt.datasets.finetuning.datasets.Math23KDataset,
    }
    experimental_datasets = {
    }
    datasets = mcpt.merge_configs(datasets, experimental_datasets)
    input_path = pathlib.Path(config['input-path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['output-path']) / config['dataset']
    return datasets[config['dataset']](
        input_path=input_path,
        output_path=output_path,
        vocab_path=config['vocab'],
        pinyin_vocab_path=config['pinyin-vocab'],
        template_id=config['template-id'],
        n_ctx=config['n-ctx'],
        use_pinyin=config['use-pinyin'],
        special_tokens=config['special-tokens'],
        split=config['split'],
        use_cache=config['use-cache'],
        items_per_file=config['items-per-file'],
        extra_config=config.get('extra-config', None),
    )
