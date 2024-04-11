import pathlib

import linglong


def load(config: dict):
    input_path = pathlib.Path(config['input_path']) / config['dataset']
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    return linglong.datasets.pretraining.PreTrainingDataset(
        linglong.datasets.pretraining.PreTrainingDatasetConfig(
            input_path=input_path,
            output_path=output_path,
            vocab_path=config['vocab_path'],
            special_tokens=config['special_tokens'],
            stride=config['stride'],
            items_per_file=config['items_per_file'],
            n_position=config['n_position'],
            use_pinyin=config['use_pinyin'],
            pinyin_vocab_path=config['pinyin_vocab_path'],
            use_cache=config['use_cache'],
        ))
