from typing import *

from mcpt.datasets.finetuning.base import BaseDataset


class CEPSUM2Dataset(BaseDataset):

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
            f'商品种类：{obj_type}；特征信息：{features}；商品描述：{"".join(obj["source"].split())}',
            [self._special_tokens['part_separator']],
            f'商品简介：{"".join(obj["target"].split())}',
        ]


class LCSTSDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'文本：{obj["text"]}',
            [self._special_tokens['part_separator']],
            f'摘要：{obj["summary"]}',
        ]


class AdGenDataset(BaseDataset):

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
            description,
            [self._special_tokens['part_separator']],
            f'商品描述：{obj["desc"]}',
        ]


class KBQADataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        a, relation, _ = obj['triple'].strip().split('|||')
        return [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            '答案：',
            a.strip(),
            [self._special_tokens['segment_separator']],
            relation.strip(),
        ]


class BaseSegmentationDataset(BaseDataset):

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_segments(obj) -> List[str]:
        raise NotImplementedError

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            [self._special_tokens['part_separator']],
            '分词结果：',
        ]
        segments = []
        for segment in self._get_segments(obj):
            segments.append(segment)
            segments.append([self._special_tokens['segment_separator']])
        # Drop the last segment separator.
        parts.extend(segments[:-1])
        return parts


class CUGESegmentationDataset(BaseSegmentationDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_segments(obj) -> List[str]:
        return obj['ans'].split()


class ICWBSegmentationDataset(BaseSegmentationDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj.replace("  ", "")

    @staticmethod
    def _get_segments(obj) -> List[str]:
        return obj.split()


class LCQMCDataset(BaseDataset):

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
            f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？',
            [self._special_tokens['part_separator']],
            ['否', '是'][obj['label']],
        ]


class Math23KDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["text"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["equation"][2:]}',
        ]
