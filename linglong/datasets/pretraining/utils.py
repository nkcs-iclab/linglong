import pathlib

from typing import *

import mcpt


def load(config: Dict[str, Any]):
    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    if config['stride'] <= 0:
        raise ValueError(f'`stride` is set to {config["stride"]}, which is not positive.')
    return mcpt.datasets.pretraining.base.FineTuningDatasetBase(
        input_path=input_path,
        output_path=output_path,
        vocab_path=config['vocab'],
        model_config=config['model_config'],
        special_tokens=config['special_tokens'],
        stride=config['stride'],
        items_per_file=config['items_per_file'],
        pinyin_vocab_path=config.get('pinyin_vocab'),
        use_cache=config['use_cache'],
        extra_config=config.get('extra_config'),
    )
