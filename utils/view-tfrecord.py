import fire
import pathlib

import mcpt
import mcpt.records
import tensorflow as tf


def main(
        path: str,
        meta: str,
        use_pinyin: bool = False,
        vocab: str = '../common/vocab/char-13312.txt',
        n_example: int = 3,
):
    meta = mcpt.load_config(meta)
    padding_shape = meta['padding_shape']
    mcpt.pprint({
        'record_count': meta['count'],
        'padding_shape': padding_shape,
        'compression_type': meta.get('compression_type'),
    })
    if use_pinyin:
        decode_fn = mcpt.records.decode_pinyin
        padded_shapes = ((2, padding_shape), padding_shape)
    else:
        decode_fn = mcpt.records.decode
        padded_shapes = (padding_shape, padding_shape)
    path = pathlib.Path(path)
    if path.is_dir():
        files = list(map(lambda x: str(path / x), meta['files']))
    else:
        files = [str(path)]
    dataset = tf.data.TFRecordDataset(files, compression_type=meta.get('compression_type'))
    dataset = dataset.map(decode_fn)
    dataset = dataset.padded_batch(n_example, padded_shapes=padded_shapes)
    tokenizer = mcpt.Tokenizer(vocab)
    for batch in dataset:
        mcpt.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
