import math
import pathlib
import tensorflow as tf

from typing import *
from torch.utils.data import IterableDataset, DataLoader

import mcpt

tf.config.set_visible_devices([], 'GPU')


def _int64_feature(value: Sequence) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(data: Sequence, label: Sequence, pinyin: Optional[Sequence] = None) -> bytes:
    feature = {
        'data': _int64_feature(data),
        'label': _int64_feature(label),
    }
    if pinyin is not None:
        feature['pinyin'] = _int64_feature(pinyin)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def decode(serialized_example: bytes) -> Tuple[tf.Tensor, tf.Tensor]:
    feature = {
        'data': tf.io.VarLenFeature(dtype=tf.int64),
        'label': tf.io.VarLenFeature(dtype=tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature)
    data = tf.sparse.to_dense(example['data'])
    label = tf.sparse.to_dense(example['label'])
    return data, label


def decode_pinyin(serialized_example: bytes) -> Tuple[tf.Tensor, tf.Tensor]:
    feature = {
        'data': tf.io.VarLenFeature(dtype=tf.int64),
        'label': tf.io.VarLenFeature(dtype=tf.int64),
        'pinyin': tf.io.VarLenFeature(dtype=tf.int64),
    }
    example = tf.io.parse_single_example(serialized_example, feature)
    data = tf.sparse.to_dense(example['data'])
    label = tf.sparse.to_dense(example['label'])
    pinyin = tf.sparse.to_dense(example['pinyin'])
    data = tf.expand_dims(data, axis=-2)
    pinyin = tf.expand_dims(pinyin, axis=-2)
    data = tf.concat((data, pinyin), axis=-2)
    return data, label


class TFRecordDataset(IterableDataset):

    def __init__(
            self,
            path: str,
            meta: str,
            dp_size: int = 1,
            dp_rank: int = 0,
            use_pinyin: bool = False,
    ):
        super().__init__()
        path = pathlib.Path(path)
        meta = mcpt.load_config(str(path / meta))
        padding_shape = meta['padding_shape']
        self.samples_per_rank = math.ceil(meta['count'] / dp_size)
        dataset = tf.data.TFRecordDataset(
            list(map(lambda x: str(path / x), meta['files'])),
            compression_type=meta.get('compression_type'),
        )

        if dp_size > 1:
            dataset = dataset.shard(dp_size, dp_rank)
        if use_pinyin:
            decode_fn = decode_pinyin
            padded_shapes = ((2, padding_shape), padding_shape)
        else:
            decode_fn = decode
            padded_shapes = (padding_shape, padding_shape)

        dataset = dataset \
            .repeat() \
            .map(decode_fn) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        if padding_shape is None:
            self.tfds = dataset.batch(1)
        else:
            self.tfds = dataset.padded_batch(1, padded_shapes=padded_shapes)
        self.index = 0
        self.tfds_iter = iter(self.tfds)

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        self.index = 0
        self.tfds_iter = iter(self.tfds)
        return self

    def __next__(self):
        if self.index >= self.samples_per_rank:
            raise StopIteration
        self.index += 1
        data, label = next(self.tfds_iter)
        return data.numpy()[0], label.numpy()[0]


def load(
        path: Optional[str],
        meta: str,
        batch_size: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        use_pinyin: bool = False,
):
    if path is None:
        return
    dataset = TFRecordDataset(path, meta, dp_size, dp_rank, use_pinyin)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
