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
            'Math23K': mcpt.datasets.evaluation.generation.Math23KDataset,
            'KBQA': mcpt.datasets.evaluation.generation.KBQADataset,
            'LCSTS': mcpt.datasets.evaluation.generation.LCSTSDataset,
            'AdGen': mcpt.datasets.evaluation.generation.AdGenDataset,
            'LCQMC': mcpt.datasets.evaluation.generation.LCQMCDataset,
            'WordSeg-Weibo': mcpt.datasets.evaluation.generation.CUGEStyleSegmentationDataset,
            'CEPSUM2-cases-bags': mcpt.datasets.evaluation.generation.CEPSUM2Dataset,
            'CEPSUM2-clothing': mcpt.datasets.evaluation.generation.CEPSUM2Dataset,
            'CEPSUM2-home-appliances': mcpt.datasets.evaluation.generation.CEPSUM2Dataset,
        },
        'perplexity': {
        }
    }
    experimental_datasets = {
        'generation': {
            'Math23KBackward': mcpt.experimental.datasets.evaluation.generation.Math23KBackwardDataset,
            'KBQABackward': mcpt.experimental.datasets.evaluation.generation.KBQABackwardDataset,
        }
    }
    datasets = mcpt.merge_configs(datasets, experimental_datasets)

    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['cache_path']) / config['dataset']

    dataset = datasets[method].get(config['dataset'], None)
    if dataset is None:
        raise ValueError(f'The {method} method is not supported for this dataset.')
    return dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        split=config['split'],
        vocab_path=config['vocab'],
        pinyin_vocab_path=config['pinyin_vocab'],
        template_id=config['template_id'],
        model_config=config['model_config'],
        special_tokens=config['special_tokens'],
        use_cache=config['use_cache'],
        extra_config=config.get('extra_config', None),
    )
