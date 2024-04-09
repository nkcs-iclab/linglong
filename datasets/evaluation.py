import fire
import numpy as np

from torch.utils.data import DataLoader

import linglong

from linglong.datasets.evaluation.base import DictDataset


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        dataset_config: str = '../evaluation/configs/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str | None = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        special_tokens: dict[str, str] | None = None,
        n_examples: int = 3,
):
    with linglong.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '[MASK]',
            'end_token': '[CLS]',
            'part_separator': '[unused1]',
            'segment_separator': '[unused2]',
            **(special_tokens or {}),
        }

        config = linglong.merge_configs({
            'dataset': dataset,
            'input_path': input_path,
            'output_path': output_path,
            'dataset_config_path': dataset_config,
            'vocab_path': vocab,
            'pinyin_vocab_path': pinyin_vocab,
            'use_cache': use_cache,
            'special_tokens': special_tokens,
        }, linglong.load_config(dataset_config, key=dataset))
        model_path = config['model'] if isinstance(config['model'], str) else config['model']['base']
        model_config = linglong.LingLongConfig.from_pretrained(model_path)
        config['use_pinyin'] = model_config.use_pinyin
        tokenizer = linglong.load_tokenizer(
            vocab_path=vocab,
            special_tokens=special_tokens,
            pretrained_model=model_path,
        )[0]
        spinner.write(linglong.prettify(config))

    with linglong.running(f'Loading {dataset} dataset', spinner=use_cache):
        dataset = linglong.datasets.evaluation.load(config)
        data, candidates = dataset.prepare()
        meta = {
            'count': len(data),
            'candidates': candidates,
        }
        spinner.write(linglong.prettify(meta))

    print(linglong.text('Examples:', style=linglong.INFO))
    dataset = DictDataset(data)
    data_loader = DataLoader(dataset, batch_size=n_examples, collate_fn=linglong.datasets.evaluation.padded_batch)
    for batch in data_loader:
        linglong.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
