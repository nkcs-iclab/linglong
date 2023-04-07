import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    if config.get('use_perplexity', False):
        method = 'perplexity'
    else:
        method = 'generation'

    datasets = {
        'generation': {
            'SIGHAN15': mcpt.datasets.evaluation.generation.SIGHANDataset,
        },
        'perplexity': {
        }
    }
    experimental_datasets = {
        'basic': {
        }
    }
    datasets = mcpt.merge_configs(datasets, experimental_datasets)

    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['cache_path']) / config['dataset']

    if (dataset := datasets[method].get(config['dataset'], None)) is None:
        raise ValueError(f'The {method} method is not supported for this dataset.')
    return dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        split=config['split'],
        vocab_path=config['vocab'],
        pinyin_vocab_path=config['pinyin_vocab'],
        template_id=config['template_id'],
        special_tokens=config['special_tokens'],
        use_cache=config['use_cache'],
        extra_config=config.get('extra_config', None),
    )
