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


class AdGenBackwardDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        description = f'标题信息：{obj["title"]}；' if obj['title'] else ''
        if 'tags' in obj:
            description += '标签信息：'
            for tag in obj['tags']:
                description += tag + '，'
            description = description[:-1] + '；'
        description += '特征信息：'
        for feature in obj['feature']:
            description += f'{feature[0]}：{feature[1]}，'
        description = description[:-1] + '；'
        return [
            description[::-1],
            [self._special_tokens['part_separator']],
            f'商品描述：{obj["desc"][::-1]}',
        ]


class LCQMCBackwardDataset(BaseDataset):

    def _load_file(self, path: str) -> List[Dict[str, Any]]:
        objs = super()._load_file(path)
        data = []
        for obj in objs:
            parts = obj.strip().split('\t')
            data.append({
                'sentence1': parts[0],
                'sentence2': parts[1],
                'label': int(parts[2]),
            })
        return data

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？'[::-1],
            [self._special_tokens['part_separator']],
            ['否', '是'][obj['label']],
        ]
