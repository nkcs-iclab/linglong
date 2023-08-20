import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    datasets = {
        'MIRACLZHQueries': mcpt.datasets.rlhf.datasets.MIRACLZHQueriesDataset,
    }
    experimental_datasets = {
        'CustomQA': mcpt.datasets.rlhf.datasets.CustomQADataset,
    }
    datasets = mcpt.merge_configs(datasets, experimental_datasets)
    name = config.get('name')
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    return datasets[config['dataset']](
        name=name,
        output_path=output_path,
        vocab_path=config['vocab'],
        stage=config['stage'],
        model_config=config['model_config'],
        special_tokens=config['special_tokens'],
        split=config['split'],
        use_cache=config['use_cache'],
        items_per_file=config['items_per_file'],
        extra_config=config.get('extra_config'),
    )
