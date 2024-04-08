import json
import math
import pathlib
import warnings

from typing import *

import mcpt


class BaseDataset:

    def __init__(
            self,
            input_path: str,
            output_path: str,
            vocab_path: str,
            template_id: int,
            model_config: Dict[str, Any],
            special_tokens: Dict[str, str],
            items_per_file: int,
            pinyin_vocab_path: Optional[str] = None,
            split: str = 'train',
            use_cache: bool = False,
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._split = split
        self._use_pinyin = model_config.get('use_pinyin', False)
        self._n_ctx = model_config['n_ctx']
        self._input_path = next(pathlib.Path(input_path).glob(f'{self._split}*'))
        self._output_path = pathlib.Path(output_path) / f'template-{template_id}{"-pinyin" if self._use_pinyin else ""}'
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._tokenizer = mcpt.Tokenizer(vocab_path)
        self._pinyin_tokenizer = mcpt.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        ) if self._use_pinyin else None
        self._template_id = template_id
        self._use_cache = use_cache
        self._items_per_file = items_per_file
        self._extra_config = extra_config
        self._file_format = None
        self._special_tokens = special_tokens

    def _load_file(self, path: str) -> Union[List, Dict]:
        return mcpt.load_file(path, format=self._file_format)

    @staticmethod
    def _discard_obj(obj, discarded: List, reason: Optional[str] = None):
        warnings.warn(f'{mcpt.pprint(obj, output_string=True)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _templatize(self, objs, i: int) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return getattr(self, f'_template_{self._template_id}')(objs[i])

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
            text = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + text
            text += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
            if self._use_pinyin:
                pinyin = self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + pinyin
                pinyin += self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
                if len(text) != len(pinyin):
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
            new_file_idx = i // self._items_per_file
            if new_file_idx != file_idx:
                if writer is not None:
                    writer.close()
                file_idx = new_file_idx
                filename = f'{self._split}-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                meta['files'].append(filename)
                writer = tf.io.TFRecordWriter(str(self._output_path / filename), options='GZIP')
            writer.write(mcpt.records.serialize_example(text, text[1:], pinyin if self._use_pinyin else None))
            meta['count'] += 1
            meta['padding_shape'] = max(len(text), meta['padding_shape'])
        meta['compression_type'] = 'GZIP'
        if writer is not None:
            writer.close()
        return meta, discarded

    def _assemble(
            self,
            parts: List[Union[str, List[str], Dict[str, List[str]]]],
    ) -> Tuple[List[int], Optional[List[int]]]:
        text = self._convert_parts_to_ids(parts)
        pinyin = self._convert_parts_to_ids(parts, use_pinyin=True) if self._use_pinyin else None
        return text, pinyin

    def _convert_parts_to_ids(
            self,
            parts: List[Union[str, List[str], Dict[str, List[str]]]],
            use_pinyin: bool = False,
    ) -> List[int]:
        tokenizer = self._pinyin_tokenizer if use_pinyin else self._tokenizer
        tokens = []
        for part in parts:
            if isinstance(part, str):
                tokens.extend(tokenizer.tokenize(part))
            elif isinstance(part, list):
                tokens.extend(part)
            elif isinstance(part, dict):
                tokens.extend(part.get('pinyin' if use_pinyin else 'text', part['text']))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def prepare(self) -> Tuple[str, str]:
        meta_path = self._output_path / f'{self._split}-meta.json'
        if not (self._use_cache and meta_path.is_file()):
            meta, discarded = self._process()
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            if len(discarded) > 0:
                print(f'\n{len(discarded)} items are discarded.')
        return str(meta_path.absolute()), str(self._output_path.absolute())
