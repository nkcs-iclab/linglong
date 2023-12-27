import abc
import pickle
import pathlib
import warnings
import numpy as np

from typing import *

import mcpt


class BaseDataset(metaclass=abc.ABCMeta):

    def __init__(
            self,
            input_path: str,
            output_path: str,
            split: str,
            vocab_path: str,
            template_id: int,
            model_config: Dict[str, Any],
            special_tokens: Dict[str, str],
            pinyin_vocab_path: Optional[str] = None,
            use_cache: bool = False,
            method: str = 'generation',
            format: str = None,
            extra_config: Optional[Dict[str, Any]] = None,
    ):
        self._input_path = input_path
        self._split = split
        self._use_pinyin = model_config.get('use_pinyin', False)
        self._tokenizer = mcpt.Tokenizer(vocab_path)
        self._pinyin_tokenizer = mcpt.PinyinTokenizer(
            vocab_file=pinyin_vocab_path,
            fallback=self._tokenizer,
        ) if self._use_pinyin else None
        self._template_id = template_id
        self._use_cache = use_cache
        self._extra_config = extra_config
        self._input_path = next(pathlib.Path(self._input_path).glob(f'{self._split}*'))
        self._output_path = pathlib.Path(output_path) / method
        self._output_path.mkdir(parents=True, exist_ok=True)

        self._candidates = None
        self._file_format = format
        self._special_tokens = special_tokens
        self._use_prompt = model_config.get('use_prompt', False)
        self._prompt_length = model_config.get('prompt_length', 0)
        self._prompt_token = self._special_tokens['prompt_token']

    def _load_file(self, path: str) -> Union[List, Dict]:
        return mcpt.load_file(path, format=self._file_format)

    @staticmethod
    def _discard_obj(obj, discarded: List):
        warnings.warn(f'The pinyin information of {mcpt.pprint(obj, export=True)} is discarded.')
        discarded.append(obj)

    # noinspection PyMethodMayBeStatic
    def _postprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return data

    def _templatize(self, objs, i: int, **kwargs) \
            -> Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        return getattr(self, f'_template_{self._template_id}')(objs[i], **kwargs)

    def _process(self) -> List[Dict[str, Any]]:
        data, discarded = [], []
        objs = self._load_file(str(self._input_path))

        for i in mcpt.trange(len(objs)):
            if self._use_prompt:
                parts, label, extra = self._templatize(objs, i, prompt_length=self._prompt_length, prompt_token=self._prompt_token)
            else:
                parts, label, extra = self._templatize(objs, i)


            text, pinyin, label = self._assemble(parts, label)
            text = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + text
            if self._use_pinyin:
                pinyin = self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + pinyin
                if len(text) != len(pinyin):
                    warnings.warn(f'`text` has size {len(text)} and `pinyin` has size {len(pinyin)}'
                                  f' (most likely due to omitted control characters).')
                    pinyin = np.zeros_like(text)
                    self._discard_obj(objs[i], discarded)
            data.append({
                'text': np.asarray([text]),
                **({'pinyin': np.asarray([pinyin])} if self._use_pinyin else {}),
                'label': np.asarray(label) if label is not None else None,
                **extra,
            })
        if len(discarded) > 0:
            warnings.warn(f'\nThe pinyin information of {len(discarded)} item(s) is discarded.')
        return data

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

    def _assemble(
            self,
            data_parts: List[Union[str, List[str], Dict[str, List[str]]]],
            label_parts: Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        label, pinyin_label = None, None
        text = self._convert_parts_to_ids(data_parts)
        pinyin = self._convert_parts_to_ids(data_parts, use_pinyin=True) if self._use_pinyin else None
        if label_parts is not None:
            label = self._convert_parts_to_ids(label_parts)
        return text, pinyin, label

    def prepare(self) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        save_path = self._output_path / f'{self._split}-template-{self._template_id}.pkl'
        if save_path.is_file() and self._use_cache:
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._process()
            data = self._postprocess(data)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        return data, self._candidates


class PerplexityDataset(BaseDataset, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(method='perplexity', **kwargs)

    def _templatize(self, objs, i: int) \
            -> Tuple[List[List[Union[str, List[str], Dict[str, List[str]]]]], Optional[int]]:
        return getattr(self, f'_template_{self._template_id}')(objs[i])

    def _process(self) -> List[Dict[str, Any]]:
        data, discarded = [], []
        objs = self._load_file(str(self._input_path))

        for i in mcpt.trange(len(objs)):
            parts_list, label = self._templatize(objs, i)
            text, label = [], [label] if label is not None else None
            pinyin = [] if self._use_pinyin else None

            for parts in parts_list:
                text_i, pinyin_i, _ = self._assemble(parts, None)
                text_i = self._tokenizer.convert_tokens_to_ids([self._special_tokens['start_token']]) + text_i
                text_i += self._tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
                text.append(np.asarray([text_i]))
                if self._use_pinyin:
                    pinyin_i = self._pinyin_tokenizer.convert_tokens_to_ids(
                        [self._special_tokens['start_token']]) + pinyin_i
                    pinyin_i += self._pinyin_tokenizer.convert_tokens_to_ids([self._special_tokens['end_token']])
                    if len(text_i) != len(pinyin_i):
                        warnings.warn(f'`text` has size {len(text_i)} and `pinyin` has size {len(pinyin_i)}.'
                                      f' (most likely due to omitted control characters).')
                        pinyin_i = np.zeros_like(text_i)
                        self._discard_obj(objs[i], discarded)
                    pinyin.append(np.asarray([pinyin_i]))
            label = np.asarray(label)
            data.append({
                'text': text,
                **({'pinyin': pinyin} if self._use_pinyin else {}),
                'label': label,
            })
        if len(discarded) > 0:
            warnings.warn(f'\nThe pinyin information of {len(discarded)} item(s) is discarded.')
        return data
