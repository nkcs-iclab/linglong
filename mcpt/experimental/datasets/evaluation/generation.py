from typing import *

from mcpt.datasets.evaluation.base import BaseDataset


class Math23KBackwardDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._candidates = [
            '%', '(', ')', '*', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '^',
        ]

    @staticmethod
    def _template_0(obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'问题：{obj["text"][::-1]}答案：',
        ]
        label = [
            obj['label'][::-1],
        ] if 'label' in obj else None
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'问题：{obj["text"][::-1]}',
            [self._special_tokens['part_separator']],
            '答案：',
        ]
        label = [
            obj['equation'][2:][::-1],
            [self._special_tokens['part_separator']],
            obj['label'][::-1],
        ] if 'label' in obj else None
        return parts, label, {}


class KBQABackwardDataset(BaseDataset):

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'问题：{obj["question"][::-1]}',
            [self._special_tokens['part_separator']],
            '答案：',
        ]
        if obj['answer']:
            a, relation, b = obj['triple'].strip().split('|||')
            label = [
                a.strip()[::-1],
                [self._special_tokens['segment_separator']],
                relation.strip()[::-1],
                [self._special_tokens['segment_separator']],
                b.strip()[::-1],
            ]
        else:
            label = None
        return parts, label, {}
