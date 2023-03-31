import abc
import pickle
import pathlib
import warnings
import numpy as np

from typing import *

import mcpt


class GenerationDataset(metaclass=abc.ABCMeta):

    def __init__(
            self,
            input_path: str,
            output_path: str,
            split: str,
            vocab_path: str,
            pinyin_vocab_path: str,
            template_id: int,
            special_tokens: Dict[str, str],
            load_from_cache: bool = False,
            method: str = 'generation',
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._input_path = input_path
        self._split = split
        self._tokenizer = mcpt.tokenization.Tokenizer(vocab_path)
        self._pinyin_tokenizer = mcpt.tokenization.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        )
        self._template_id = template_id
        self._load_from_cache = load_from_cache
        self._extra_config = extra_config
        self._input_path = next(pathlib.Path(self._input_path).glob(f'{self._split}*'))
        self._output_path = pathlib.Path(output_path) / method
        self._output_path.mkdir(parents=True, exist_ok=True)

        self._candidates = None
        self._file_format = None
        self._templates = {}
        self._special_tokens = special_tokens

    def raw_data(self) -> List[Dict[str, Any]]:
        return self._load_file(str(self._input_path))

    def _load_file(self, path: str):
        return mcpt.load_file(path, format=self._file_format)

    @staticmethod
    def _discard_obj(obj, discarded: List):
        warnings.warn(f'The pinyin information of {mcpt.print_dict(obj, export=True)} is discarded.')
        discarded.append(obj)

    def _postprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return data

    def _templatize(self, objs, i: int) \
            -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        return self._templates[self._template_id](objs[i])

    def _process(self) -> List[Dict[str, Any]]:
        data, discarded = [], []
        objs = self._load_file(str(self._input_path))

        for i in mcpt.trange(len(objs)):
            parts, label, extra = self._templatize(objs, i)
            text, pinyin, label, _ = self._assemble(parts, label)
            text = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start-token']]) + text
            pinyin = self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['start-token']]) + pinyin
            if len(text) != len(pinyin):
                warnings.warn(f'`text` has size {len(text)} and `pinyin` has size {len(pinyin)}'
                              f' (most likely due to omitted control characters).')
                pinyin = np.zeros_like(text)
                self._discard_obj(objs[i], discarded)
            data.append({
                'text': np.asarray([text]),
                'pinyin': np.asarray([pinyin]),
                'label': np.asarray(label) if label is not None else None,
                **extra,
            })
        if len(discarded) > 0:
            warnings.warn(f'\nThe pinyin information of {len(discarded)} item(s) is discarded.')
        return data

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

    def _assemble(self, data_parts: List[Dict[str, Any]], label_parts: Optional[List[Dict[str, Any]]]) \
            -> Tuple[List[int], List[int], Optional[List[int]], Optional[List[int]]]:
        label, pinyin_label = None, None
        text = self._convert_parts_to_ids(data_parts)
        pinyin = self._convert_parts_to_ids(data_parts, use_pinyin=True)
        if label_parts is not None:
            label = self._convert_parts_to_ids(label_parts)
            pinyin_label = self._convert_parts_to_ids(label_parts, use_pinyin=True)
        return text, pinyin, label, pinyin_label

    def prepare(self) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        save_path = self._output_path / f'{self._split}-template-{self._template_id}.pkl'
        if save_path.is_file() and self._load_from_cache:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._process()
            data = self._postprocess(data)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        return data, self._candidates


class PerplexityDataset(GenerationDataset, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(method='perplexity', **kwargs)

    def _templatize(self, objs, i: int) -> Tuple[List[List[Dict[str, Any]]], Optional[int]]:
        return self._templates[self._template_id](objs[i])

    def _process(self) -> List[Dict[str, Any]]:
        data, discarded = [], []
        objs = self._load_file(str(self._input_path))

        for i in mcpt.trange(len(objs)):
            parts_list, label = self._templatize(objs, i)
            text, pinyin, label = [], [], [label] if label is not None else None

            for parts in parts_list:
                text_i, pinyin_i, _, _ = self._assemble(parts, None)
                text_i = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start-token']]) + text_i
                text_i += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end-token']])
                pinyin_i = self._pinyin_tokenizer.convert_tokens_to_ids(
                    [self._special_tokens['start-token']]) + pinyin_i
                pinyin_i += self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['end-token']])
                if len(text_i) != len(pinyin_i):
                    warnings.warn(f'`text` has size {len(text_i)} and `pinyin` has size {len(pinyin_i)}.'
                                  f' (most likely due to omitted control characters).')
                    pinyin_i = np.zeros_like(text_i)
                    self._discard_obj(objs[i], discarded)
                text.append(np.asarray([text_i]))
                pinyin.append(np.asarray([pinyin_i]))
            label = np.asarray(label)
            data.append({
                'text': text,
                'pinyin': pinyin,
                'label': label,
            })
        if len(discarded) > 0:
            warnings.warn(f'\nThe pinyin information of {len(discarded)} item(s) is discarded.')
        return data
