import pathlib

import linglong

from linglong.datasets.pretraining.base import PreTrainingDatasetConfig


def load(config: dict):
    input_path = pathlib.Path(config['input_path']) / config['dataset']
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    if config['stride'] <= 0:
        raise ValueError(f'`stride` is set to {config["stride"]}, which is not positive.')
    return linglong.datasets.pretraining.PreTrainingDataset(
        PreTrainingDatasetConfig(
            input_path=input_path,
            output_path=output_path,
            vocab_path=config['vocab_path'],
            special_tokens=config['special_tokens'],
            stride=config['stride'],
            items_per_file=config['items_per_file'],
            n_positions=config['n_positions'],
            use_pinyin=config['use_pinyin'],
            pinyin_vocab_path=config['pinyin_vocab_path'],
            use_cache=config['use_cache'],
            extra_config=config.get('extra_config'),
        ))
