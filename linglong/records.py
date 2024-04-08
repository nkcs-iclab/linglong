import torch
import pathlib
import tensorflow as tf

from typing import Sequence
from torch.utils.data import IterableDataset

import linglong

tf.config.set_visible_devices([], 'GPU')


def _int64_feature(value: Sequence) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(data: Sequence, pinyin: Sequence | None = None, attention_mask: Sequence | None = None) -> bytes:
    feature = {
        'data': _int64_feature(data),
    }
    if pinyin is not None:
        feature['pinyin'] = _int64_feature(pinyin)
    if attention_mask is not None:
        feature['attention_mask'] = _int64_feature(attention_mask)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def get_decode_fn(
        padding_shape: int | None = None,
        use_pinyin: bool = False,
        load_attention_mask: bool = True,
        ignore_index: int = -100,
):
    def decode(serialized_example: bytes) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        feature = {
            'data': tf.io.VarLenFeature(dtype=tf.int64),
        }
        if load_attention_mask:
            feature['attention_mask'] = tf.io.VarLenFeature(dtype=tf.int64)
        example = tf.io.parse_single_example(serialized_example, feature)

        data = tf.sparse.to_dense(example['data'])
        attention_mask = tf.sparse.to_dense(example['attention_mask']) if load_attention_mask else tf.ones_like(data)

        padded_data = tf.pad(data, [[0, padding_shape - tf.shape(data)[0]]], constant_values=0)
        padded_attention_mask = tf.pad(
            attention_mask,
            [[0, padding_shape - tf.shape(attention_mask)[0]]],
            constant_values=0,
        )
        padded_label = tf.pad(data, [[0, padding_shape - tf.shape(data)[0]]], constant_values=ignore_index)

        return padded_data, padded_attention_mask, padded_label

    def decode_pinyin(serialized_example: bytes) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        feature = {
            'data': tf.io.VarLenFeature(dtype=tf.int64),
            'pinyin': tf.io.VarLenFeature(dtype=tf.int64),
        }
        if load_attention_mask:
            feature['attention_mask'] = tf.io.VarLenFeature(dtype=tf.int64)
        example = tf.io.parse_single_example(serialized_example, feature)

        data = tf.sparse.to_dense(example['data'])
        pinyin = tf.sparse.to_dense(example['pinyin'])
        attention_mask = tf.sparse.to_dense(example['attention_mask']) if load_attention_mask else tf.ones_like(data)

        padded_data = tf.pad(data, [[0, padding_shape - tf.shape(data)[0]]], constant_values=0)
        padded_pinyin = tf.pad(pinyin, [[0, padding_shape - tf.shape(pinyin)[0]]], constant_values=0)
        padded_attention_mask = tf.pad(
            attention_mask,
            [[0, padding_shape - tf.shape(attention_mask)[0]]],
            constant_values=0,
        )
        padded_label = tf.pad(data, [[0, padding_shape - tf.shape(data)[0]]], constant_values=ignore_index)

        return padded_data, padded_pinyin, padded_attention_mask, padded_label

    return decode_pinyin if use_pinyin else decode


class TFRecordDataset(IterableDataset):

    def __init__(
            self,
            path: str,
            meta: str,
            files_key: str = 'files',
            use_pinyin: bool = False,
            load_attention_mask: bool = True,
    ):
        super().__init__()
        path = pathlib.Path(path)
        meta = linglong.load_config(str(path / meta))
        padding_shape = meta['padding_shape']
        self.count = meta['count']
        self.use_pinyin = use_pinyin
        dataset = tf.data.TFRecordDataset(
            list(map(lambda x: str(path / x), meta[files_key])) if path.is_dir() else str(path),
            compression_type=meta.get('compression_type'),
        )
        decode_fn = get_decode_fn(
            padding_shape=padding_shape,
            use_pinyin=self.use_pinyin,
            load_attention_mask=load_attention_mask,
        )
        dataset = dataset \
            .repeat() \
            .map(decode_fn) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.dataset = dataset
        self.index = 0
        self.dataset_iter = iter(self.dataset)

    def __len__(self):
        return self.count

    def __iter__(self):
        self.index = 0
        self.dataset_iter = iter(self.dataset)
        return self

    def __next__(self):
        if self.index >= self.count:
            raise StopIteration
        self.index += 1
        if self.use_pinyin:
            data, pinyin, attention_mask, label = next(self.dataset_iter)
            return {
                'input_ids': torch.from_numpy(data.numpy()).long(),
                'pinyin_input_ids': torch.from_numpy(pinyin.numpy()).long(),
                'attention_mask': torch.from_numpy(attention_mask.numpy()).long(),
                'label_ids': torch.from_numpy(label.numpy()).long(),
            }
        # noinspection PyTupleAssignmentBalance
        data, attention_mask, label = next(self.dataset_iter)
        return {
            'input_ids': torch.from_numpy(data.numpy()).long(),
            'attention_mask': torch.from_numpy(attention_mask.numpy()).long(),
            'label_ids': torch.from_numpy(label.numpy()).long(),
        }


def load(
        path: str | None,
        meta: str,
        use_pinyin: bool = False,
        load_attention_mask: bool = True,
) -> TFRecordDataset | None:
    if path is None:
        return
    return TFRecordDataset(path, meta, use_pinyin=use_pinyin, load_attention_mask=load_attention_mask)
