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
