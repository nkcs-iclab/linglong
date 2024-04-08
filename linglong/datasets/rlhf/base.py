import json
import math
import pathlib
import warnings

from typing import *

import mcpt


class BaseDataset:

    def __init__(
            self,
            output_path: str,
            vocab_path: str,
            stage: int,
            model_config: Dict[str, Any],
            special_tokens: Dict[str, str],
            items_per_file: int,
            name: Optional[str] = None,
            split: str = 'train',
            use_cache: bool = False,
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._split = split
        self._n_ctx = model_config['n_ctx']
        self._output_path = pathlib.Path(output_path) / f'stage-{stage}'
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._tokenizer = mcpt.Tokenizer(vocab_path)
        self._stage = stage
        self._use_cache = use_cache
        self._items_per_file = items_per_file
        self._extra_config = extra_config
        self._special_tokens = special_tokens
        self._name = name or self._dataset_name()

    def _dataset_name(self) -> str:
        raise ValueError('This dataset has no default name. Please specify the name.')

    def _load_dataset(self, path: str) -> Union[List, Dict]:
        raise NotImplementedError

    def _chosen_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        raise NotImplementedError('This dataset does not contain chosen parts.')

    def _rejected_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        raise NotImplementedError('This dataset does not contain rejected parts.')

    def _prompt_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        raise NotImplementedError('This dataset does not contain prompt parts.')

    @staticmethod
    def _discard_obj(obj, discarded: List, reason: Optional[str] = None):
        warnings.warn(f'{mcpt.pprint(obj, output_string=True)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _process(self) -> Tuple[Dict[str, Any], List]:
        objs = self._load_dataset(self._name)
        discarded = []
        chosen_writer, rejected_writer, prompt_writer = None, None, None
        file_idx = None
        n_file = math.ceil(len(objs) / self._items_per_file)

        meta = {
            'padding_shape': 0,
            'count': 0,
        }
        if self._stage == 1 or self._stage == 2:
            meta['chosen_files'] = []
        if self._stage == 2:
            meta['rejected_files'] = []
        if self._stage == 3:
            meta['prompt_files'] = []

        import mcpt.records
        import tensorflow as tf
        for i in mcpt.trange(len(objs)):
            if self._stage == 1 or self._stage == 2:
                chosen_parts = self._chosen_template(objs[i])
                chosen_ids = self._convert_parts_to_ids(chosen_parts)
                chosen_ids = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + chosen_ids
                chosen_ids += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
                if len(chosen_ids) > self._n_ctx:
                    self._discard_obj(
                        objs[i],
                        discarded,
                        f'`chosen_ids` has size {len(chosen_ids)}, exceeding `n_ctx`: {self._n_ctx}.',
                    )
                    continue
                rejected_ids = []
            if self._stage == 2:
                rejected_parts = self._rejected_template(objs[i])
                rejected_ids = self._convert_parts_to_ids(rejected_parts)
                rejected_ids = self._tokenizer.convert_tokens_to_ids(
                    [self._special_tokens['start_token']]) + rejected_ids
                rejected_ids += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
                if len(rejected_ids) > self._n_ctx:
                    self._discard_obj(
                        objs[i],
                        discarded,
                        f'`rejected_ids` has size {len(rejected_ids)}, exceeding `n_ctx`: {self._n_ctx}.',
                    )
                    continue
            if self._stage == 3:
                prompt_parts = self._prompt_template(objs[i])
                prompt_ids = self._convert_parts_to_ids(prompt_parts)
                prompt_ids = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + prompt_ids
                if len(prompt_ids) > self._n_ctx:
                    self._discard_obj(
                        objs[i],
                        discarded,
                        f'`prompt_ids` has size {len(prompt_ids)}, exceeding `n_ctx`: {self._n_ctx}.',
                    )
                    continue

            new_file_idx = i // self._items_per_file
            if new_file_idx != file_idx:
                if chosen_writer is not None:
                    chosen_writer.close()
                if rejected_writer is not None:
                    rejected_writer.close()
                if prompt_writer is not None:
                    prompt_writer.close()
                file_idx = new_file_idx
                chosen_filename = f'{self._split}-chosen-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                rejected_filename = f'{self._split}-rejected-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                prompt_filename = f'{self._split}-prompt-{file_idx + 1:0{len(str(n_file))}d}-of-{n_file}.tfrecord.gz'
                if self._stage == 1 or self._stage == 2:
                    meta['chosen_files'].append(chosen_filename)
                    chosen_writer = tf.io.TFRecordWriter(str(self._output_path / chosen_filename), options='GZIP')
                if self._stage == 2:
                    meta['rejected_files'].append(rejected_filename)
                    rejected_writer = tf.io.TFRecordWriter(str(self._output_path / rejected_filename), options='GZIP')
                if self._stage == 3:
                    meta['prompt_files'].append(prompt_filename)
                    prompt_writer = tf.io.TFRecordWriter(str(self._output_path / prompt_filename), options='GZIP')

            if self._stage == 1 or self._stage == 2:
                chosen_writer.write(mcpt.records.serialize_example(chosen_ids, chosen_ids[1:], None))
                meta['padding_shape'] = max(len(chosen_ids), meta['padding_shape'])
            if self._stage == 2:
                rejected_writer.write(mcpt.records.serialize_example(rejected_ids, rejected_ids[1:], None))
                meta['padding_shape'] = max(len(chosen_ids), len(rejected_ids), meta['padding_shape'])
            if self._stage == 3:
                prompt_writer.write(mcpt.records.serialize_example(prompt_ids, prompt_ids[1:], None))
                meta['padding_shape'] = max(len(prompt_ids), meta['padding_shape'])
            meta['count'] += 1
        meta['compression_type'] = 'GZIP'
        if chosen_writer is not None:
            chosen_writer.close()
        if rejected_writer is not None:
            rejected_writer.close()
        if prompt_writer is not None:
            prompt_writer.close()
        return meta, discarded

    def _convert_parts_to_ids(
            self,
            parts: List[Union[str, List[str], Dict[str, List[str]]]],
    ) -> List[int]:
        tokens = []
        for part in parts:
            if isinstance(part, str):
                tokens.extend(self._tokenizer.tokenize(part))
            elif isinstance(part, list):
                tokens.extend(part)
            elif isinstance(part, dict):
                tokens.extend(part['text'])
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
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
