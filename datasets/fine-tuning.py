import fire
import pickle
import pathlib

from typing import *

import mcpt
import mcpt.records
import tensorflow as tf


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        split: str = 'train',
        n_ctx: int = 1024,
        dataset_config: str = 'configs/fine-tuning/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str = '../common/vocab/pinyin-1354.txt',
        use_pinyin: bool = False,
        use_cache: bool = False,
        items_per_file: int = 200000,
        special_tokens: Optional[Dict[str, str]] = None,
        n_example: int = 3,
):
    with mcpt.running('Loading configs') as spinner:
        special_tokens = {
            'start-token': '[MASK]',
            'end-token': '[CLS]',
            'part-separator': '[unused1]',
            'segment-separator': '[unused2]',
            **(special_tokens or {}),
        }
        config = mcpt.merge_configs({
            'dataset': dataset,
            'dataset-config': dataset_config,
            'input-path': input_path,
            'output-path': output_path,
            'split': split,
            'n-ctx': n_ctx,
            'vocab': vocab,
            'pinyin-vocab': pinyin_vocab,
            'use-pinyin': use_pinyin,
            'use-cache': use_cache,
            'items-per-file': items_per_file,
            'special-tokens': special_tokens,
        }, mcpt.load_config(dataset_config, key=dataset))
        dataset_path = pathlib.Path(output_path) / dataset
        spinner.write(mcpt.print_dict(config, export=True))

    with mcpt.running(f'Processing {dataset} dataset', spinner=use_cache) as spinner:
        dataset = mcpt.datasets.finetuning.load(config)
        meta_path, records_path = dataset.prepare()
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        padding_shape = meta['padding_shape']
        spinner.write(mcpt.print_dict({
            'meta': meta_path,
            'records': records_path,
            'record_count': meta['count'],
            'padding_shape': padding_shape,
        }, export=True))

    print(mcpt.text('Examples:', style=mcpt.INFO))
    if use_pinyin:
        dataset_path = dataset_path / f'template-{config["template-id"]}-pinyin'
        decode_fn = mcpt.records.decode_pinyin
        padded_shapes = ((2, padding_shape), padding_shape, padding_shape)
    else:
        dataset_path = dataset_path / f'template-{config["template-id"]}'
        decode_fn = mcpt.records.decode
        padded_shapes = (padding_shape, padding_shape, padding_shape)

    dataset = tf.data.TFRecordDataset(list(map(lambda x: str(dataset_path / x), meta['files'])))
    dataset = dataset.map(decode_fn)
    dataset = dataset.padded_batch(n_example, padded_shapes=padded_shapes)
    tokenizer = mcpt.Tokenizer(vocab)
    for batch in dataset:
        mcpt.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
