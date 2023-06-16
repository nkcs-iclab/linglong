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

    def _load_file(self, path: str) -> Union[List, Dict]:
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


class Math23KBackwardDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["text"][::-1]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["equation"][2:][::-1]}',
        ]


class BaseSegmentationBackwardDataset(BaseDataset):

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_segments(obj) -> List[str]:
        raise NotImplementedError

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        parts = [
            f'原始文本：{self._get_text(obj)[::-1]}',
            [self._special_tokens['part_separator']],
            '分词结果：',
        ]
        segments = []
        for segment in self._get_segments(obj):
            segments.append(segment[::-1])
            segments.append([self._special_tokens['segment_separator']])
        # Drop the last segment separator.
        parts.extend(segments[:-1][::-1])
        return parts


class CUGEStyleSegmentationBackwardDataset(BaseSegmentationBackwardDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_segments(obj) -> List[str]:
        return obj['ans'].split()


class ICWBSegmentationBackwardDataset(BaseSegmentationBackwardDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj.replace("  ", "")

    @staticmethod
    def _get_segments(obj) -> List[str]:
        return obj.split()


class CEPSUM2BackwardDataset(BaseDataset):

    def _load_file(self, path: str) -> Union[List, Dict]:
        objs = super()._load_file(path)
        data = []
        for obj in objs:
            for target in obj['tgt']:
                data.append({
                    'feature': obj['kb'],
                    'type': obj['type'],
                    'target': target,
                    'source': obj['src'],
                })
        return data

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        obj_type = {
            'bc': '箱包',
            'cl': '衣服',
            'homea': '家具',
        }[obj['type']]

        features = '；'.join(
            [f'{"".join(k.split())}：{"".join(v.split())}' for k, v in obj['feature'].items()]
        )
        return [
            f'商品种类：{obj_type[::-1]}；特征信息：{features[::-1]}；商品描述：{"".join(obj["source"].split())[::-1]}',
            [self._special_tokens['part_separator']],
            f'商品简介：{"".join(obj["target"].split())[::-1]}',
        ]
