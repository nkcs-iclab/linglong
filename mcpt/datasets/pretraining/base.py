import json
import pathlib
import warnings

from typing import *

import mcpt


class FileLoader:

    def __init__(
            self,
            file_list: List[pathlib.Path],
            tokenizer: 'mcpt.Tokenizer',
            special_tokens: Dict[str, str],
    ):
        self.file_list = file_list
        self._tokenizer = tokenizer
        self._special_tokens = special_tokens
        self.progbar = mcpt.tqdm(total=len(file_list))
        self._current_file_idx = 0
        self._text_pool = []

    def empty(self) -> bool:
        return self._current_file_idx >= len(self.file_list)

    def load(self, length: int, stride: int) -> Tuple[List[str], List[str]]:
        while len(self._text_pool) < length + 1 and self._current_file_idx < len(self.file_list):
            with open(self.file_list[self._current_file_idx], 'r') as f:
                self._current_file_idx += 1
                self.progbar.update(1)
                self._text_pool.append(self._special_tokens['start_token'])
                for line in f:
                    self._text_pool.extend(self._tokenizer.tokenize(line.strip()))
                self._text_pool.append(self._special_tokens['end_token'])
        text = self._text_pool[:length]
        label = self._text_pool[1:length + 1]
        self._text_pool = self._text_pool[stride:]
        return text, label


class BaseDataset:

    def __init__(
            self,
            input_path: str,
            output_path: str,
            vocab_path: str,
            model_config: Dict[str, Any],
            special_tokens: Dict[str, str],
            stride: int,
            items_per_file: int,
            pinyin_vocab_path: Optional[str] = None,
            use_cache: bool = False,
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._use_pinyin = model_config.get('use_pinyin', False)
        self._n_ctx = model_config['n_ctx']
        self._input_file_list = pathlib.Path(input_path).glob(f'**/*.txt')
        self._output_path = pathlib.Path(output_path) / f'template-0{"-pinyin" if self._use_pinyin else ""}'
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._tokenizer = mcpt.Tokenizer(vocab_path)
        self._pinyin_tokenizer = mcpt.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        ) if self._use_pinyin else None
        self._use_cache = use_cache
        self._stride = stride
        self._items_per_file = items_per_file
        self._extra_config = extra_config
        self._file_format = None
        self._special_tokens = special_tokens

    @staticmethod
    def _discard_obj(obj, discarded: List, reason: Optional[str] = None):
        warnings.warn(f'{mcpt.pprint(obj, export=True)} is discarded. Reason: {reason}')
        warnings.warn(f'{len(discarded)} items are discarded.')
        discarded.append(obj)

    def _process(self) -> Tuple[Dict[str, Any], List]:
        file_loader = FileLoader(
            list(self._input_file_list),
            special_tokens=self._special_tokens,
            tokenizer=self._tokenizer,
        )
        discarded = []
        writer = None
        file_idx = None
        meta = {
            'padding_shape': self._n_ctx,
            'count': 0,
            'files': [],
            'compression_type': 'GZIP',
        }
        item_id = 0
        import mcpt.records
        import tensorflow as tf
        while not file_loader.empty():
            text, label = file_loader.load(length=self._n_ctx, stride=self._stride)
            text_id = self._tokenizer.convert_tokens_to_ids(text)
            label_id = self._tokenizer.convert_tokens_to_ids(label)
            pinyin_id = None
            if self._use_pinyin:
                pinyin_id = self._pinyin_tokenizer.convert_tokenizer_tokens_to_ids(text)
                if len(text_id) != len(pinyin_id):
                    self._discard_obj(
                        text,
                        discarded,
                        reason=f'`text_id` and `pinyin_id` have different lengths: {len(text_id)} vs {len(pinyin_id)}.'
                               f' (most likely due to omitted control characters).',
                    )
                    continue
            new_file_idx = item_id // self._items_per_file
            if new_file_idx != file_idx:
                if writer is not None:
                    writer.close()
                file_idx = new_file_idx
                filename = f'{file_idx + 1:08d}.tfrecord.gz'
                meta['files'].append(filename)
                writer = tf.io.TFRecordWriter(str(self._output_path / filename), options='GZIP')
            writer.write(mcpt.records.serialize_example(text_id, label_id, pinyin_id if self._use_pinyin else None))
            meta['count'] += 1
            item_id += 1
        if writer is not None:
            writer.close()
        return meta, discarded

    def prepare(self) -> Tuple[str, str]:
        meta_path = self._output_path / f'train-meta.json'
        if not (self._use_cache and meta_path.is_file()):
            meta, discarded = self._process()
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            if len(discarded) > 0:
                print(f'\n{len(discarded)} items are discarded.')
        return str(meta_path.absolute()), str(self._output_path.absolute())
