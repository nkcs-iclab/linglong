import torch
import pathlib

import linglong

from linglong.datasets.evaluation.base import EvaluationDatasetConfig


def load(config: dict):
    datasets = {
        'CEPSUM2-cases-bags': linglong.datasets.evaluation.CEPSUM2Dataset,
        'CEPSUM2-clothing': linglong.datasets.evaluation.CEPSUM2Dataset,
        'CEPSUM2-home-appliances': linglong.datasets.evaluation.CEPSUM2Dataset,
        'LCSTS': linglong.datasets.evaluation.LCSTSDataset,
        'AdGen': linglong.datasets.evaluation.AdGenDataset,
        'KBQA': linglong.datasets.evaluation.KBQADataset,
        'WordSeg-Weibo': linglong.datasets.evaluation.CUGESegmentationDatasetBase,
        'ICWB-MSR': linglong.datasets.evaluation.ICWBSegmentationDatasetBase,
        'LCQMC': linglong.datasets.evaluation.LCQMCDataset,
        'Math23K': linglong.datasets.evaluation.Math23KDataset,
        'SIGHAN15': linglong.datasets.evaluation.SIGHANDataset,
        'CMeEE': linglong.datasets.evaluation.CMeEEDataset,
    }
    experimental_datasets = {}
    datasets = linglong.merge_configs(datasets, experimental_datasets)
    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    return datasets[config['dataset']](
        EvaluationDatasetConfig(
            input_path=input_path,
            output_path=output_path,
            model_path=config['model'] if isinstance(config['model'], str) else config['model']['base'],
            vocab_path=config['vocab_path'],
            pinyin_vocab_path=config['pinyin_vocab_path'],
            template_id=config['template_id'],
            special_tokens=config['special_tokens'],
            use_pinyin=config['use_pinyin'],
            split=config['split'],
            use_cache=config['use_cache'],
            extra_config=config.get('extra_config'),
        )
    )


def pad_sequence(
        sequences,
        batch_first: bool = False,
        padding_value: float = 0.0,
        padding_side: str = 'right',
):
    if padding_side == 'right':
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    elif padding_side == 'left':
        sequences = list(map(lambda s: s.flip(0), sequences))
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    else:
        raise ValueError(f'`padding_side` should be either "right" or "left", but got {padding_side}')
    return padded_sequence


def padded_batch(batch):
    output = {}
    keys = batch[0].keys()
    for k in keys:
        if k == 'label_ids':
            padding_side = 'right'
            padding_value = -100
        else:
            padding_side = 'left'
            padding_value = 0
        output[k] = pad_sequence(
            [x[k] for x in batch],
            batch_first=True,
            padding_value=padding_value,
            padding_side=padding_side,
        )
    return output
