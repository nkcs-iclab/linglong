from typing import *

from mcpt.datasets.evaluation.base import BaseDataset


class SIGHANDataset(BaseDataset):

    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        objs = super()._load_file(path)
        return list(objs.values())

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        source = obj['text']
        target = list(source)
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            f'原始文本：{source}',
            [self._special_tokens['part_separator']],
            '纠错后文本：',
        ]
        label = [
            ''.join(target),
        ] if len(obj['errors']) > 0 else None
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        source = obj['text']
        target = list(source)
        corrections = []
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            corrections.append(f'{error_index}:-{target[error_index]}+{error[1]}')
        parts = [
            f'原始文本：{source}',
            [self._special_tokens['part_separator']],
            '纠错：',
        ]
        label = [
            ';'.join(corrections),
        ] if len(obj['errors']) > 0 else None
        return parts, label, {}

    def _template_2(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        target = list(obj['text'])
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            obj['text'],
            [self._special_tokens['end_token']],
        ]
        label = [
            [self._special_tokens['start_token']],
            ''.join(target),
            [self._special_tokens['end_token']],
        ]
        return parts, label, {}
