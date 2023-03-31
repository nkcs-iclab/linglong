import math
import pickle
import pathlib
import warnings
import numpy as np

from typing import *

import mcpt


class BaseDataset:

    def __init__(
            self,
            input_path: str,
            output_path: str,
            vocab_path: str,
            pinyin_vocab_path: str,
            template_id: int,
            n_ctx: int,
            use_pinyin: bool,
            special_tokens: Dict[str, str],
            items_per_file: int,
            split: str = 'train',
            use_cache: bool = False,
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._split = split
        self._use_pinyin = use_pinyin
        self._input_path = next(pathlib.Path(input_path).glob(f'{self._split}*'))
        self._output_path = pathlib.Path(output_path) / f'template-{template_id}{"-pinyin" if self._use_pinyin else ""}'
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._tokenizer = mcpt.tokenization.Tokenizer(vocab_path)
        self._pinyin_tokenizer = mcpt.tokenization.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        )
        self._template_id = template_id
        self._n_ctx = n_ctx
        self._use_cache = use_cache
        self._items_per_file = items_per_file
        self._extra_config = extra_config

        self._templates = {}
        self._file_format = None
        self._special_tokens = special_tokens

    def _load_file(self, path: str):
        return mcpt.load_file(path, format=self._file_format)

    @staticmethod
    def _discard_obj(obj, discarded: List, reason: str = ''):
        warnings.warn(f'{mcpt.print_dict(obj, export=True)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _templatize(self, objs, i: int) -> List[Dict[str, Any]]:
        return self._templates[self._template_id](objs[i])

    def _process(self) -> Tuple[Dict[str, Any], List]:
        objs = self._load_file(str(self._input_path))
        discarded = []
        writer = None
        file_idx = None
        n_file = math.ceil(len(objs) / self._items_per_file)
        meta = {
            'padding_shape': 0,
            'count': 0,
            'files': [],
        }
        import mcpt.records
        import tensorflow as tf
        for i in mcpt.trange(len(objs)):
            parts = self._templatize(objs, i)
            text, pinyin = self._assemble(parts)
            text = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start-token']]) + text
            text += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end-token']])
            if self._use_pinyin:
                pinyin = self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['start-token']]) + pinyin
                pinyin += self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['end-token']])
                if self._use_pinyin and len(text) != len(pinyin):
                    self._discard_obj(
                        objs[i],
                        discarded,
                        reason=f'`text` and `pinyin` have different lengths: {len(text)} vs {len(pinyin)}.'
                               f' (most likely due to omitted control characters).',
                    )
                    continue
            if len(text) > self._n_ctx:
                self._discard_obj(objs[i], discarded, f'`text` has size {len(text)}, exceeding `n_ctx`: {self._n_ctx}.')
                continue
            if (file_idx_ := i // self._items_per_file) != file_idx:
                if writer is not None:
                    writer.close()
                file_idx = file_idx_
                filename = f'{self._split}-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord'
                meta['files'].append(filename)
                writer = tf.io.TFRecordWriter(str(self._output_path / filename))
            writer.write(
                mcpt.records.serialize_example(
                    text,
                    text[1:],
                    np.ones_like(text[1:]),
                    pinyin if self._use_pinyin else None,
                )
            )
            meta['count'] += 1
            meta['padding_shape'] = max(len(text), meta['padding_shape'])
        if writer is not None:
            writer.close()
        return meta, discarded

    def _assemble(self, parts: List[Dict[str, Any]]) -> Tuple[List[int], Optional[List[int]]]:
        text = self._convert_parts_to_ids(parts)
        pinyin = self._convert_parts_to_ids(parts, use_pinyin=True) if self._use_pinyin else None
        return text, pinyin

    def _convert_parts_to_ids(self, parts: List[Dict[str, Any]], use_pinyin: bool = False) -> List[int]:
        tokenizer = self._pinyin_tokenizer if use_pinyin else self._tokenizer
        tokens = []
        for part in parts:
            if isinstance(part['text'], list):
                tokens.extend(part.get('pinyin' if use_pinyin else 'text', part['text']))
            else:
                tokens.extend(tokenizer.tokenize(part['text']))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def prepare(self) -> Tuple[str, str]:
        meta_path = self._output_path / f'{self._split}-meta.pkl'
        if not (self._use_cache and meta_path.is_file()):
            meta, discarded = self._process()
            with open(meta_path, 'wb') as f:
                pickle.dump(meta, f)
            if len(discarded) > 0:
                print(f'\n{len(discarded)} items are discarded.')
        return str(meta_path.absolute()), str(self._output_path.absolute())
