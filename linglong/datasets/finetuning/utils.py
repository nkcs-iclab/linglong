import pathlib

import linglong

from linglong.datasets.finetuning.base import FineTuningDatasetConfig


def load(config: dict):
    datasets = {
        'CEPSUM2-cases-bags': linglong.datasets.finetuning.CEPSUM2Dataset,
        'CEPSUM2-clothing': linglong.datasets.finetuning.CEPSUM2Dataset,
        'CEPSUM2-home-appliances': linglong.datasets.finetuning.CEPSUM2Dataset,
        'LCSTS': linglong.datasets.finetuning.LCSTSDataset,
        'AdGen': linglong.datasets.finetuning.AdGenDataset,
        'KBQA': linglong.datasets.finetuning.KBQADataset,
        'WordSeg-Weibo': linglong.datasets.finetuning.CUGESegmentationDataset,
        'ICWB-MSR': linglong.datasets.finetuning.ICWBSegmentationDataset,
        'LCQMC': linglong.datasets.finetuning.LCQMCDataset,
        'Math23K': linglong.datasets.finetuning.Math23KDataset,
        'CMeEE': linglong.datasets.finetuning.CMeEEDataset,
    }
    experimental_datasets = {}
    datasets = linglong.merge_configs(datasets, experimental_datasets)
    input_path = pathlib.Path(config['input_path']) / config.get('base', config['dataset'])
    output_path = pathlib.Path(config['output_path']) / config['dataset']
    return datasets[config['dataset']](
        FineTuningDatasetConfig(
            input_path=input_path,
            output_path=output_path,
            vocab_path=config['vocab'],
            pinyin_vocab_path=config['pinyin_vocab'],
            template_id=config['template_id'],
            special_tokens=config['special_tokens'],
            items_per_file=config['items_per_file'],
            n_positions=config['n_positions'],
            use_pinyin=config['use_pinyin'],
            split=config['split'],
            use_cache=config['use_cache'],
            extra_config=config.get('extra_config'),
        )
    )
