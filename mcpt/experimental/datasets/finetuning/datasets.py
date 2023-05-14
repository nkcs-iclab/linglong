from typing import *

from mcpt.datasets.finetuning.base import BaseDataset


class CustomQADataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["answer"]}',
        ]


class CustomMathDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["answer"]}',
            [self._special_tokens['part_separator']],
            f'分析：{obj["analysis"]}',
        ]


class KBQABackwardDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        a, relation, _ = obj['triple'].strip().split('|||')
        return [
            f'问题：{obj["question"][::-1]}',
            [self._special_tokens['part_separator']],
            '答案：',
            relation.strip()[::-1],
            [self._special_tokens['segment_separator']],
            a.strip()[::-1],
        ]


class LCSTSBackwardDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'文本：{obj["text"][::-1]}',
            [self._special_tokens['part_separator']],
            f'摘要：{obj["summary"][::-1]}',
        ]
