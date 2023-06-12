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


class LCSTSBackwardDataset(BaseDataset):

    @staticmethod
    def _template_0(obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'文本：{obj["text"][::-1]}摘要：',
        ]
        label = [
            obj['summary'][::-1],
        ] if obj['summary'] else None
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'文本：{obj["text"][::-1]}',
            [self._special_tokens['part_separator']],
            '摘要：',
        ]
        label = [
            obj['summary'][::-1],
        ] if obj['summary'] else None
        return parts, label, {}


class AdGenBackwardDataset(BaseDataset):

    @staticmethod
    def _template_0(obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        description = f'标题信息：{obj["title"]}；' if obj['title'] else ''
        if 'tags' in obj:
            description += '标签信息：'
            for tag in obj['tags']:
                description += tag + '，'
            description = description[:-1] + '；'
        description += '特征信息：'
        for feature in obj['feature']:
            description += f'{feature[0]}：{feature[1]}，'
        description = description[:-1] + '；商品描述：'
        parts = [
            description[::-1],
        ]
        label = [
            obj['desc'][::-1],
        ]
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
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
        parts = [
            description[::-1],
            [self._special_tokens['part_separator']],
            '商品描述：',
        ]
        label = [
            obj['desc'][::-1],
        ]
        return parts, label, {}


class LCQMCBackwardDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._candidates = ['是', '否']

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

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'“{obj["sentence1"]}”与“{obj["sentence2"]}”的意思是否矛盾？'[::-1],
        ]
        label = [
            self._candidates[obj['label']],
        ]
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？'[::-1],
            [self._special_tokens['part_separator']],
        ]
        label = [
            self._candidates[1 - obj['label']],
        ]
        return parts, label, {}


class BaseSegmentationBackwardDataset(BaseDataset):

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_segments(obj) -> List[str]:
        raise NotImplementedError

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'原始文本：{self._get_text(obj)}'[::-1],
            [self._special_tokens['part_separator']],
            '分词结果：',
        ]
        label = []
        for segment in self._get_segments(obj):
            label.append(segment[::-1])
            label.append([self._special_tokens['segment_separator']])
        label = label[:-1][::-1]
        return parts, label, {}


class CUGEStyleSegmentationBackwardDataset(BaseSegmentationBackwardDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_segments(obj) -> List[str]:
        return obj['ans'].split()


class CEPSUM2BackwardDataset(BaseDataset):

    def _load_file(self, path: str) -> Union[List, Dict]:
        objs = super()._load_file(path)
        data = []
        for obj in objs:
            if obj['tgt'] == '':
                obj['tgt'] = ['']
            for target in obj['tgt']:
                data.append({
                    'feature': obj['kb'],
                    'type': obj['type'],
                    'target': target,
                    'source': obj['src'],
                    'id': obj['idx'],
                })
        return data

    def _postprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups = {}
        for obj in data:
            if f'{obj["type"]}-{obj["id"]}' in groups:
                if groups[f'{obj["type"]}-{obj["id"]}']['label'] is not None:
                    groups[f'{obj["type"]}-{obj["id"]}']['label'].append(obj['label'])
            else:
                obj['label'] = [obj['label']] if obj['label'] is not None else None
                groups[f'{obj["type"]}-{obj["id"]}'] = obj
        return list(groups.values())

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        obj_type = {
            'bc': '箱包',
            'cl': '衣服',
            'homea': '家具',
        }[obj['type']]
        features = '；'.join(
            [f'{"".join(k.split())}：{"".join(v.split())}' for k, v in obj['feature'].items()]
        )
        parts = [
            f'商品种类：{obj_type[::-1]}；特征信息：{features[::-1]}；商品描述：{"".join(obj["source"].split())[::-1]}',
            [self._special_tokens['part_separator']],
            '商品简介：',
        ]
        label = [
            "".join(obj["target"].split())[::-1],
        ] if obj['target'] else None
        return parts, label, {'id': obj['id'], 'type': obj['type']}
