from typing import *

from mcpt.datasets.evaluation.base import GenerationDataset


class SIGHANDataset(GenerationDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._templates = {
            0: self._template_0,
            1: self._template_1,
            2: self._template_2,
        }
        self._file_format = 'json'

    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        objs = super()._load_file(path)
        return list(objs.values())

    def _template_0(self, obj) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        source = obj['text']
        target = list(source)
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            {'text': f'原始文本：{source}'},
            {'text': [self._special_tokens['part-separator']]},
            {'text': '纠错后文本：'},
        ]
        label = [
            {'text': ''.join(target)},
        ] if len(obj['errors']) > 0 else None
        return parts, label, {}

    def _template_1(self, obj) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        source = obj['text']
        target = list(source)
        corrections = []
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            corrections.append(f'{error_index}:-{target[error_index]}+{error[1]}')
        parts = [
            {'text': f'原始文本：{source}'},
            {'text': [self._special_tokens['part-separator']]},
            {'text': '纠错：'}
        ]
        label = [
            {'text': ';'.join(corrections)},
        ] if len(obj['errors']) > 0 else None
        return parts, label, {}

    def _template_2(self, obj) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        target = list(obj['text'])
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            {'text': obj['text']},
            {'text': [self._special_tokens['end-token']]},
        ]
        label = [
            {'text': [self._special_tokens['start-token']]},
            {'text': ''.join(target)},
            {'text': [self._special_tokens['end-token']]},
        ]
        return parts, label, {}
