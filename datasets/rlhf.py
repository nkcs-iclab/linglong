import fire
import pathlib

from typing import *

import mcpt
import mcpt.records
import tensorflow as tf


def print_examples_from_dataset(
        path: pathlib.Path,
        meta: Dict,
        vocab: str,
        files_key: str = 'files',
        n_example: int = 3,
):
    dataset = tf.data.TFRecordDataset(
        list(map(lambda x: str(path / x), meta[files_key])),
        compression_type=meta.get('compression_type'),
    )
    dataset = dataset.map(mcpt.records.decode)
    dataset = dataset.padded_batch(n_example, padded_shapes=(meta['padding_shape'], meta['padding_shape']))
    tokenizer = mcpt.Tokenizer(vocab)
    for batch in dataset:
        mcpt.print_training_records(batch, tokenizer=tokenizer)
        break


def main(
        dataset: str,
        output_path: str,
        model_config: str,
        stage: int,
        split: str = 'train',
        dataset_config: str = 'configs/rlhf/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        use_cache: bool = False,
        items_per_file: int = 200000,
        special_tokens: Optional[Dict[str, str]] = None,
        n_example: int = 3,
):
    with mcpt.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '[MASK]',
            'end_token': '[CLS]',
            'part_separator': '[unused1]',
            'segment_separator': '[unused2]',
            **(special_tokens or {}),
        }
        model_config_path = model_config
        model_config = mcpt.load_config(model_config_path)
        config = mcpt.merge_configs({
            'dataset': dataset,
            'dataset_config_path': dataset_config,
            'model_config_path': model_config_path,
            'model_config': model_config,
            'stage': stage,
            'output_path': output_path,
            'split': split,
            'vocab': vocab,
            'use_cache': use_cache,
            'items_per_file': items_per_file,
            'special_tokens': special_tokens,
        }, mcpt.load_config(dataset_config, key=dataset) or {})
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running(f'Processing {dataset} dataset', spinner=use_cache) as spinner:
        dataset = mcpt.datasets.rlhf.load(config)
        meta_path, records_path = dataset.prepare()
        meta = mcpt.load_config(meta_path)
        spinner.write(mcpt.pprint({
            'meta': meta_path,
            'records': records_path,
            'record_count': meta['count'],
            'padding_shape': meta['padding_shape'],
        }, export=True))

    print(mcpt.text('Chosen examples:', style=mcpt.INFO))
    print_examples_from_dataset(
        path=pathlib.Path(records_path),
        meta=meta,
        vocab=vocab,
        files_key='chosen_files',
        n_example=n_example,
    )
    if stage == 2:
        print(mcpt.text('Rejected examples:', style=mcpt.INFO))
        print_examples_from_dataset(
            path=pathlib.Path(records_path),
            meta=meta,
            vocab=vocab,
            files_key='rejected_files',
            n_example=n_example,
        )


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
