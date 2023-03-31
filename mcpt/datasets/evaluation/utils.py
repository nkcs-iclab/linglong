import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    if config.get('use-perplexity', False):
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

    input_path = pathlib.Path(config['input-path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['cache-path']) / config['dataset']

    if (dataset := datasets[method].get(config['dataset'], None)) is None:
        raise ValueError(f'The {method} method is not supported for this dataset.')
    return dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        split=config['split'],
        vocab_path=config['vocab'],
        pinyin_vocab_path=config['pinyin-vocab'],
        template_id=config['template-id'],
        special_tokens=config['special-tokens'],
        use_cache=config['use-cache'],
        extra_config=config.get('extra-config', None),
    )
