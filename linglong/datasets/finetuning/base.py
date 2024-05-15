import json
import math
import pathlib
import warnings
import dataclasses

import linglong


@dataclasses.dataclass
class FineTuningDatasetConfig:
    input_path: str | pathlib.Path
    output_path: str | pathlib.Path
    template_id: int
    special_tokens: dict[str, str]
    items_per_file: int
    n_position: int
    use_pinyin: bool = False
    vocab_path: str | None = None
    pinyin_vocab_path: str | None = None
    split: str = 'train'
    use_cache: bool = False
    model_path: str | None = None
    extra_config: dict | None = None

    def __post_init__(self):
        self.input_path = pathlib.Path(self.input_path)
        self.output_path = pathlib.Path(
            self.output_path) / f'template-{self.template_id}{"-pinyin" if self.use_pinyin else ""}'


class FineTuningDatasetBase:

    def __init__(self, config: FineTuningDatasetConfig):
        self.config = config
        self.input_file = next(self.config.input_path.glob(f'{self.config.split}*'))
        self.tokenizer = linglong.get_tokenizers(
            vocab_path=self.config.vocab_path,
            pinyin_vocab_path=self.config.pinyin_vocab_path,
            pretrained_model=self.config.model_path,
            special_tokens=self.config.special_tokens,
            use_pinyin=self.config.use_pinyin,
        )
        self.pinyin_tokenizer = None
        if self.config.use_pinyin:
            self.tokenizer, self.pinyin_tokenizer = self.tokenizer
        self.file_format = None

        self.config.output_path.mkdir(parents=True, exist_ok=True)

    def _load_file(self, path: str) -> list | dict:
        return linglong.load_file(path, format=self.file_format)

    @staticmethod
    def _discard_obj(obj, discarded: list, reason: str | None = None):
        warnings.warn(f'{linglong.prettify(obj)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _templatize(self, obj) -> list:
        parts = getattr(self, f'_template_{self.config.template_id}')(obj)
        if not (all(isinstance(part, tuple) for part in parts) or all(not isinstance(part, tuple) for part in parts)):
            raise ValueError('All parts should be tuples or none of them should be.')
        return parts

    def _prepend_start_token(
            self,
            parts: list,
            is_label: bool = False,
    ) -> list:
        if parts and isinstance(parts[0], tuple):
            parts.insert(0, (self.tokenizer.bos_token, is_label))
        else:
            parts.insert(0, self.tokenizer.bos_token)
        return parts

    def _append_end_token(
            self,
            parts: list,
            is_label: bool = True,
    ) -> list:
        if parts and isinstance(parts[0], tuple):
            parts.append((self.tokenizer.eos_token, is_label))
        else:
            parts.append(self.tokenizer.eos_token)
        return parts

    def _add_start_and_end_tokens(self, parts: list) -> list:
        self._prepend_start_token(parts)
        self._append_end_token(parts)
        return parts

    def _process(self) -> tuple[dict, list]:
        objs = self._load_file(str(self.input_file))
        discarded = []
        writer = None
        file_idx = None
        n_file = math.ceil(len(objs) / self.config.items_per_file)
        meta = {
            'padding_shape': 0,
            'count': 0,
            'files': [],
            'has_attention_mask': False,
        }
        import linglong.data.tfrecord
        import tensorflow as tf
        for i in linglong.trange(len(objs)):
            parts = self._templatize(objs[i])
            input_ids, pinyin_input_ids, label_ids = self._encode(parts)
            if self.config.use_pinyin and len(input_ids) != len(pinyin_input_ids):
                self._discard_obj(
                    objs[i],
                    discarded,
                    reason=f'`input_ids` and `pinyin_input_ids` have different lengths: '
                           f'{len(input_ids)} vs {len(pinyin_input_ids)}. '
                           f'(most likely due to omitted control characters.)',
                )
                continue
            if len(input_ids) > self.config.n_position:
                self._discard_obj(
                    objs[i],
                    discarded,
                    f'`input_ids` has size {len(input_ids)}, '
                    f'exceeding `n_position`: {self.config.n_position}.',
                )
                continue
            new_file_idx = i // self.config.items_per_file
            if new_file_idx != file_idx:
                if writer is not None:
                    writer.close()
                file_idx = new_file_idx
                filename = f'{self.config.split}-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                meta['files'].append(filename)
                writer = tf.io.TFRecordWriter(str(self.config.output_path / filename), options='GZIP')
            writer.write(linglong.data.tfrecord.serialize_example(
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

    def _encode(self, parts: list) -> tuple[list[int], list[int] | None, list[int] | None]:
        input_ids, label_ids = self._convert_parts_to_ids(parts=parts)
        pinyin_input_ids = self._convert_parts_to_ids(
            parts=parts,
            use_pinyin=True,
            output_labels=False,
        ) if self.config.use_pinyin else None
        return input_ids, pinyin_input_ids, label_ids

    def _convert_parts_to_ids(
            self,
            parts: list,
            use_pinyin: bool = False,
            output_labels: bool = True,
            ignore_index: int = -100,
    ) -> list[int] | tuple[list[int], list[int]]:
        tokenizer = self.pinyin_tokenizer if use_pinyin else self.tokenizer
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
        meta_path = self.config.output_path / f'{self.config.split}-meta.json'
        if not (self.config.use_cache and meta_path.is_file()):
            meta, discarded = self._process()
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            if len(discarded) > 0:
                print(f'\n{len(discarded)} items are discarded.')
        return str(meta_path.absolute()), str(self.config.output_path.absolute())
