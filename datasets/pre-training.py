import fire
import pathlib

import linglong
import linglong.data.tfrecord
import tensorflow as tf


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        model_config: str,
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        stride: int | None = None,
        items_per_file: int = 200000,
        special_tokens: dict[str, str] | None = None,
        n_example: int = 3,
):
    with linglong.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '<|startoftext|>',
            'end_token': '<|endoftext|>',
            'part_separator': '<unused1>',
            'segment_separator': '<unused2>',
            **(special_tokens or {}),
        }
        model_config_path = model_config
        model_config = linglong.LingLongConfig.from_json_file(model_config_path)
        config = {
            'dataset': dataset,
            'input_path': input_path,
            'output_path': output_path,
            'model_config_path': model_config_path,
            'model_config': model_config,
            'vocab_path': vocab,
            'pinyin_vocab_path': pinyin_vocab,
            'use_cache': use_cache,
            'stride': stride or model_config['n_ctx'] // 2,
            'items_per_file': items_per_file,
            'special_tokens': special_tokens,
            'use_pinyin': model_config.use_pinyin,
            'n_positions': model_config.n_positions,
        }
        spinner.write(linglong.prettify(config))

    with linglong.running(f'Processing {dataset} dataset', spinner=use_cache) as spinner:
        dataset = mcpt.datasets.pretraining.load(config)
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
        dataset_path = dataset_path / f'template-0-pinyin'
        decode_fn = mcpt.records.decode_pinyin
        padded_shapes = ((2, padding_shape), padding_shape)
    else:
        dataset_path = dataset_path / f'template-0'
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
