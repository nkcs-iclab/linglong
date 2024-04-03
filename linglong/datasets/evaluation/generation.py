import re

from typing import *

from mcpt.datasets.evaluation.base import BaseDataset


class CEPSUM2Dataset(BaseDataset):

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
            f'商品种类：{obj_type}；特征信息：{features}；商品描述：{"".join(obj["source"].split())}',
            [self._special_tokens['part_separator']],
            '商品简介：',
        ]
        label = [
            "".join(obj["target"].split()),
        ] if obj['target'] else None
        return parts, label, {'id': obj['id'], 'type': obj['type']}


class LCSTSDataset(BaseDataset):

    @staticmethod
    def _template_0(obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'文本：{obj["text"]}摘要：',
        ]
        label = [
            obj['summary'],
        ] if obj['summary'] else None
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'文本：{obj["text"]}',
            [self._special_tokens['part_separator']],
            '摘要：',
        ]
        label = [
            obj['summary'],
        ] if obj['summary'] else None
        return parts, label, {}


class AdGenDataset(BaseDataset):

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
            description,
        ]
        label = [
            obj['desc'],
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
            description,
            [self._special_tokens['part_separator']],
            '商品描述：',
        ]
        label = [
            obj['desc'],
        ]
        return parts, label, {}


class KBQADataset(BaseDataset):

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'问题：{obj["question"]}',
            [self._special_tokens['part_separator']],
            '答案：',
        ]
        if obj['answer']:
            a, relation, b = obj['triple'].strip().split('|||')
            label = [
                a.strip(),
                [self._special_tokens['segment_separator']],
                relation.strip(),
                [self._special_tokens['segment_separator']],
                b.strip(),
            ]
        else:
            label = None
        return parts, label, {}


class BaseSegmentationDataset(BaseDataset):

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
            f'原始文本：{self._get_text(obj)}',
            [self._special_tokens['part_separator']],
            '分词结果：',
        ]
        label = []
        for segment in self._get_segments(obj):
            label.append(segment)
            label.append([self._special_tokens['segment_separator']])
        label = label[:-1]
        return parts, label, {}


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
            f'“{obj["sentence1"]}”与“{obj["sentence2"]}”的意思是否矛盾？',
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
            f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？',
            [self._special_tokens['part_separator']],
        ]
        label = [
            self._candidates[1 - obj['label']],
        ]
        return parts, label, {}


class Math23KDataset(BaseDataset):

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
            f'问题：{obj["text"]}答案：',
        ]
        label = [
            obj['label'],
        ] if 'label' in obj else None
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'问题：{obj["text"]}',
            [self._special_tokens['part_separator']],
            '答案：',
        ]
        label = [
            obj['equation'][2:],
            [self._special_tokens['part_separator']],
            obj['label'],
        ] if 'label' in obj else None
        return parts, label, {}


class SIGHANDataset(BaseDataset):

    def _load_file(self, path: str) -> Union[List, Dict]:
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


class BaseNERDataset(BaseDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entity_types = {}

    def entity_type(self, entity_type: str) -> str:
        return self._entity_types[entity_type]

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_entities(obj) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _has_label(obj) -> bool:
        raise NotImplementedError

    def _template_0(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            [self._special_tokens['part_separator']],
            '实体：',
        ]
        label = None
        if self._has_label(obj):
            label = []
            for entity in self._get_entities(obj):
                label.append(f'{self.entity_type(entity["type"])}：{entity["entity"]}')
                label.append([self._special_tokens['segment_separator']])
            label = label[:-1]
        return parts, label, {}

    def _template_1(self, obj) -> \
            Tuple[
                List[Union[str, List[str], Dict[str, List[str]]]],
                Optional[List[Union[str, List[str], Dict[str, List[str]]]]],
                Dict[str, Any],
            ]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            [self._special_tokens['part_separator']],
            '实体：',
        ]
        label = None
        if self._has_label(obj):
            label = []
            entities = {
                entity['entity']: self.entity_type(entity['type'])
                for entity in self._get_entities(obj)
            }
            pattern = rf'({"|".join([re.escape(_) for _ in entities.keys()])})'
            text_splits = re.split(pattern, self._get_text(obj))
            for text_split in text_splits:
                if text_split in entities:
                    label.extend([
                        [self._special_tokens['entity_prefix']],
                        f'{entities[text_split]}：{text_split}',
                        [self._special_tokens['entity_postfix']],
                    ])
                else:
                    label.append(text_split)
        return parts, label, {}


class CUGENERDataset(BaseNERDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_entities(obj) -> List[Dict[str, Any]]:
        return obj['entities']

    @staticmethod
    def _has_label(obj) -> bool:
        return 'entities' in obj


class CMeEEDataset(CUGENERDataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entity_types = {
            'dis': '疾病',
            'sym': '临床表现',
            'dru': '药物',
            'equ': '医疗设备',
            'pro': '医疗程序',
            'bod': '身体',
            'ite': '医学检验项目',
            'mic': '微生物类',
            'dep': '科室',
        }
