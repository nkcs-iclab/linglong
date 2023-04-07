from typing import *

from mcpt.datasets.finetuning.base import BaseDataset


class Math23KDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._templates = {
            0: self._template_0,
        }
        self._file_format = 'json'

    def _template_0(self, obj) -> List[Dict[str, Any]]:
        return [
            {'text': f'问题：{obj["text"]}'},
            {'text': [self._special_tokens['part_separator']]},
            {'text': f'答案：{obj["equation"][2:]}'},
        ]


class CustomQADataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._templates = {
            0: self._template_0,
        }
        self._file_format = 'jsonl'

    def _template_0(self, obj) -> List[Dict[str, Any]]:
        return [
            {'text': f'问题：{obj["question"]}'},
            {'text': [self._special_tokens['part_separator']]},
            {'text': f'答案：{obj["answer"]}'}
        ]
