import zhconv

from typing import *
from datasets import load_dataset

import mcpt.utils

from mcpt.datasets.rlhf.base import BaseDataset


class MIRACLZHQueriesDataset(BaseDataset):

    def _dataset_name(self) -> str:
        return 'Cohere/miracl-zh-queries-22-12'

    def _load_dataset(self, name: str) -> Union[List, Dict]:
        dataset = load_dataset(name, split=self._split)
        objs = []
        for obj in dataset:
            query = obj['query']
            positive_passages = obj['positive_passages']
            negative_passages = obj['negative_passages']
            if self._stage == 1 or self._stage == 3:
                objs.append({
                    'prompt': zhconv.convert(query, 'zh-hans'),
                    'chosen': zhconv.convert(positive_passages[0]['text'], 'zh-hans'),
                })
            elif self._stage == 2:
                for positive_passage in positive_passages:
                    for negative_passage in negative_passages:
                        objs.append({
                            'prompt': zhconv.convert(query, 'zh-hans'),
                            'chosen': zhconv.convert(positive_passage['text'], 'zh-hans'),
                            'rejected': zhconv.convert(negative_passage['text'], 'zh-hans'),
                        })
        return objs

    def _chosen_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["prompt"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["chosen"]}',
        ]

    def _rejected_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["prompt"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["rejected"]}',
        ]

    def _prompt_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["prompt"]}',
            [self._special_tokens['part_separator']],
            f'答案：',
        ]


class CustomQADataset(BaseDataset):

    def _load_dataset(self, path: str) -> Union[List, Dict]:
        return mcpt.utils.load_file(path)

    def _chosen_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["answer"]}',
        ]

    def _prompt_template(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            f'答案：',
        ]
