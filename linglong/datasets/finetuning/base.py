import json
import math
import pathlib
import warnings

import linglong


class BaseDataset:

    def __init__(
            self,
            input_path: str,
            output_path: str,
            vocab_path: str,
            template_id: int,
            model_config: linglong.LingLongConfig,
            special_tokens: dict[str, str],
            items_per_file: int,
            pinyin_vocab_path: str | None = None,
            split: str = 'train',
            use_cache: bool = False,
            extra_config: dict | None = None,
    ):
        self._split = split
        self._use_pinyin = model_config.use_pinyin
        self._n_positions = model_config.n_positions
        self._input_path = next(pathlib.Path(input_path).glob(f'{self._split}*'))
        self._output_path = pathlib.Path(output_path) / f'template-{template_id}{"-pinyin" if self._use_pinyin else ""}'
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._tokenizer = linglong.Tokenizer(vocab_path)
        self._pinyin_tokenizer = linglong.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        ) if self._use_pinyin else None
        self._template_id = template_id
        self._use_cache = use_cache
        self._items_per_file = items_per_file
        self._extra_config = extra_config
        self._file_format = None
        self._special_tokens = special_tokens

    def _load_file(self, path: str) -> list | dict:
        return linglong.load_file(path, format=self._file_format)

    @staticmethod
    def _discard_obj(obj, discarded: list, reason: str | None = None):
        warnings.warn(f'{linglong.prettify(obj)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _templatize(self, obj) -> list:
        parts = getattr(self, f'_template_{self._template_id}')(obj)
        if not (all(isinstance(part, tuple) for part in parts) or all(not isinstance(part, tuple) for part in parts)):
            raise ValueError('All parts should be tuples or none of them should be.')
        return parts

    def _prepend_start_token(
            self,
            parts: list,
            is_label: bool = False,
    ):
        if parts and isinstance(parts[0], tuple):
            parts.insert(0, (self._special_tokens['start_token'], is_label))
        else:
            parts.insert(0, self._special_tokens['start_token'])

    def _append_end_token(
            self,
            parts: list,
            is_label: bool = True,
    ):
        if parts and isinstance(parts[0], tuple):
            parts.append((self._special_tokens['end_token'], is_label))
        else:
            parts.append(self._special_tokens['end_token'])

    def _process(self) -> tuple[dict, list]:
        objs = self._load_file(str(self._input_path))
        discarded = []
        writer = None
        file_idx = None
        n_file = math.ceil(len(objs) / self._items_per_file)
        meta = {
            'padding_shape': 0,
            'count': 0,
            'files': [],
            'has_attention_mask': False,
        }
        import linglong.records
        import tensorflow as tf
        for i in linglong.trange(len(objs)):
            parts = self._templatize(objs[i])
            input_ids, pinyin_input_ids, label_ids = self._encode(parts)
            if self._use_pinyin and len(input_ids) != len(pinyin_input_ids):
                self._discard_obj(
                    objs[i],
                    discarded,
                    reason=f'`input_ids` and `pinyin_input_ids` have different lengths: '
                           f'{len(input_ids)} vs {len(pinyin_input_ids)}. '
                           f'(most likely due to omitted control characters.)',
                )
                continue
            if len(input_ids) > self._n_positions:
                self._discard_obj(
                    objs[i],
                    discarded,
                    f'`input_ids` has size {len(input_ids)}, exceeding `n_positions`: {self._n_positions}.',
                )
                continue
            new_file_idx = i // self._items_per_file
            if new_file_idx != file_idx:
                if writer is not None:
                    writer.close()
                file_idx = new_file_idx
                filename = f'{self._split}-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                meta['files'].append(filename)
                writer = tf.io.TFRecordWriter(str(self._output_path / filename), options='GZIP')
            writer.write(linglong.records.serialize_example(
                data=input_ids,
                pinyin=pinyin_input_ids,
                label=label_ids,
            ))
            meta['count'] += 1
            meta['has_label'] = label_ids is not None
            meta['padding_shape'] = max(len(input_ids), meta['padding_shape'])
        meta['compression_type'] = 'GZIP'
        if writer is not None:
            writer.close()
        return meta, discarded

    def _encode(
            self,
            parts: list,
    ) -> tuple[list[int], list[int] | None, list[int] | None]:
        input_ids, label_ids = self._convert_parts_to_ids(parts=parts)
        pinyin_input_ids = self._convert_parts_to_ids(
            parts=parts,
            use_pinyin=True,
            output_labels=False,
        ) if self._use_pinyin else None
        return input_ids, pinyin_input_ids, label_ids

    def _convert_parts_to_ids(
            self,
            parts: list,
            use_pinyin: bool = False,
            output_labels: bool = True,
            ignore_index: int = -100,
    ) -> list[int] | tuple[list[int], list[int]]:
        tokenizer = self._pinyin_tokenizer if use_pinyin else self._tokenizer
        has_label = parts and isinstance(parts[0], tuple)
        input_ids = []
        label_ids = [] if output_labels and has_label else None
        for part in parts:
            if has_label:
                part, is_label = part
            else:
                is_label = True
            if isinstance(part, str):
                part_ids = tokenizer.encode(part)
            elif isinstance(part, list):
                part_ids = tokenizer.convert_tokens_to_ids(part)
            elif isinstance(part, dict):
                part_ids = tokenizer.encode(part.get('pinyin' if use_pinyin else 'text', part['text']))
            else:
                raise ValueError(f'Unsupported part type: {type(part)}.')
            input_ids.extend(part_ids)
            if has_label:
                label_ids.extend(part_ids if is_label else [ignore_index] * len(part_ids))
        if output_labels:
            return input_ids, label_ids
        return input_ids

    def prepare(self) -> tuple[str, str]:
        meta_path = self._output_path / f'{self._split}-meta.json'
        if not (self._use_cache and meta_path.is_file()):
            meta, discarded = self._process()
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            if len(discarded) > 0:
                print(f'\n{len(discarded)} items are discarded.')
        return str(meta_path.absolute()), str(self._output_path.absolute())
