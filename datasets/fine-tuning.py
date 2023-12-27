import fire
import pathlib

from typing import *

import mcpt
import mcpt.records
import tensorflow as tf


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        model_config: str,
        split: str = 'train',
        dataset_config: str = 'configs/fine-tuning/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        items_per_file: int = 200000,
        special_tokens: Optional[Dict[str, str]] = None,
        n_example: int = 3,
        format: Optional[str] = None,
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
            'input_path': input_path,
            'output_path': output_path,
            'split': split,
            'vocab': vocab,
            'pinyin_vocab': pinyin_vocab,
            'use_cache': use_cache,
            'items_per_file': items_per_file,
            'special_tokens': special_tokens,
            'format': format,
        }, mcpt.load_config(dataset_config, key=dataset))
        dataset_path = pathlib.Path(output_path) / dataset
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running(f'Processing {dataset} dataset', spinner=use_cache) as spinner:
        dataset = mcpt.datasets.finetuning.load(config)
        meta_path, records_path = dataset.prepare()
        meta = mcpt.load_config(meta_path)
        padding_shape = meta['padding_shape']
        spinner.write(mcpt.pprint({
            'meta': meta_path,
            'records': records_path,
            'record_count': meta['count'],
            'padding_shape': padding_shape,
        }, export=True))

    print(mcpt.text('Examples:', style=mcpt.INFO))
    if model_config.get('use_pinyin', False):
        dataset_path = dataset_path / f'template-{config["template_id"]}-pinyin'
        decode_fn = mcpt.records.decode_pinyin
        padded_shapes = ((2, padding_shape), padding_shape)
    else:
        dataset_path = dataset_path / f'template-{config["template_id"]}'
        decode_fn = mcpt.records.decode
        padded_shapes = (padding_shape, padding_shape)

    dataset = tf.data.TFRecordDataset(
        list(map(lambda x: str(dataset_path / x), meta['files'])),
        compression_type=meta.get('compression_type'),
    )
    dataset = dataset.map(decode_fn)
    dataset = dataset.padded_batch(n_example, padded_shapes=padded_shapes)
    tokenizer = mcpt.Tokenizer(vocab)
    for batch in dataset:
        mcpt.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
