import re

from linglong.datasets.evaluation.base import EvaluationDatasetBase


class CEPSUM2Dataset(EvaluationDatasetBase):

    def _load_file(self, path: str) -> list | dict:
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

    def _post_process(self, data: list[dict]) -> list[dict]:
        groups = {}
        for obj in data:
            if f'{obj["type"]}-{obj["id"]}' in groups:
                if groups[f'{obj["type"]}-{obj["id"]}']['label_ids'] is not None:
                    groups[f'{obj["type"]}-{obj["id"]}']['label_ids'].append(obj['label_ids'])
            else:
                obj['label_ids'] = [obj['label_ids']] if obj['label_ids'] is not None else None
                groups[f'{obj["type"]}-{obj["id"]}'] = obj
        return list(groups.values())

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
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
            self.config.special_tokens['part_separator'],
            '商品简介：',
        ]
        label = [
            "".join(obj["target"].split()),
        ] if obj['target'] else None
        return self._prepend_start_token(parts), label, {'id': obj['id'], 'type': obj['type']}


class LCSTSDataset(EvaluationDatasetBase):

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'文本：{obj["text"]}摘要：',
        ]
        label = [
            obj['summary'],
        ] if obj['summary'] else None
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'文本：{obj["text"]}',
            self.config.special_tokens['part_separator'],
            '摘要：',
        ]
        label = [
            obj['summary'],
        ] if obj['summary'] else None
        return self._prepend_start_token(parts), label


class AdGenDataset(EvaluationDatasetBase):

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
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
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
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
            self.config.special_tokens['part_separator'],
            '商品描述：',
        ]
        label = [
            obj['desc'],
        ]
        return self._prepend_start_token(parts), label


class KBQADataset(EvaluationDatasetBase):

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'问题：{obj["question"]}',
            self.config.special_tokens['part_separator'],
            '答案：',
        ]
        if obj['answer']:
            a, relation, b = obj['triple'].strip().split('|||')
            label = [
                a.strip(),
                self.config.special_tokens['segment_separator'],
                relation.strip(),
                self.config.special_tokens['segment_separator'],
                b.strip(),
            ]
        else:
            label = None
        return self._prepend_start_token(parts), label


class SegmentationDatasetBase(EvaluationDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_segments(obj) -> list[str]:
        raise NotImplementedError

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            self.config.special_tokens['part_separator'],
            '分词结果：',
        ]
        label = []
        for segment in self._get_segments(obj):
            label.append(segment)
            label.append(self.config.special_tokens['segment_separator'])
        label = label[:-1]
        return self._prepend_start_token(parts), label


class CUGESegmentationDatasetBase(SegmentationDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_segments(obj) -> list[str]:
        return obj['ans'].split()


class ICWBSegmentationDatasetBase(SegmentationDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        return obj.replace("  ", "")

    @staticmethod
    def _get_segments(obj) -> list[str]:
        return obj.split()


class LCQMCDataset(EvaluationDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = ['是', '否']

    def _load_file(self, path: str) -> list | dict:
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

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'“{obj["sentence1"]}”与“{obj["sentence2"]}”的意思是否矛盾？',
        ]
        label = [
            self.candidates[obj['label']],
        ]
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？',
            self.config.special_tokens['part_separator'],
        ]
        label = [
            self.candidates[1 - obj['label']],
        ]
        return self._prepend_start_token(parts), label


class Math23KDataset(EvaluationDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = [
            '%', '(', ')', '*', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', '^',
        ]

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'问题：{obj["text"]}答案：',
        ]
        label = [
            obj['label'],
        ] if 'label' in obj else None
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'问题：{obj["text"]}',
            self.config.special_tokens['part_separator'],
            '答案：',
        ]
        label = [
            obj['equation'][2:],
            self.config.special_tokens['part_separator'],
            obj['label'],
        ] if 'label' in obj else None
        return self._prepend_start_token(parts), label


class SIGHANDataset(EvaluationDatasetBase):

    def _load_file(self, path: str) -> list | dict:
        objs = super()._load_file(path)
        return list(objs.values())

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        source = obj['text']
        target = list(source)
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            f'原始文本：{source}',
            self.config.special_tokens['part_separator'],
            '纠错后文本：',
        ]
        label = [
            ''.join(target),
        ] if len(obj['errors']) > 0 else None
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        source = obj['text']
        target = list(source)
        corrections = []
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            corrections.append(f'{error_index}:-{target[error_index]}+{error[1]}')
        parts = [
            f'原始文本：{source}',
            self.config.special_tokens['part_separator'],
            '纠错：',
        ]
        label = [
            ';'.join(corrections),
        ] if len(obj['errors']) > 0 else None
        return self._prepend_start_token(parts), label

    def _template_2(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        target = list(obj['text'])
        for error in obj['errors']:
            error_index = int(error[0]) - 1
            correct_char = error[1]
            target[error_index] = correct_char
        parts = [
            obj['text'],
        ]
        label = [
            ''.join(target),
        ]
        return self._add_start_and_end_tokens(parts), self._add_start_and_end_tokens(label)


class BaseNERDataset(EvaluationDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_types = {}

    def entity_type(self, entity_type: str) -> str:
        return self.entity_types[entity_type]

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_entities(obj) -> list[dict]:
        raise NotImplementedError

    @staticmethod
    def _has_label(obj) -> bool:
        raise NotImplementedError

    def _template_0(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            self.config.special_tokens['part_separator'],
            '实体：',
        ]
        label = None
        if self._has_label(obj):
            label = []
            for entity in self._get_entities(obj):
                label.append(f'{self.entity_type(entity["type"])}：{entity["entity"]}')
                label.append(self.config.special_tokens['segment_separator'])
            label = label[:-1]
        return self._prepend_start_token(parts), label

    def _template_1(self, obj) -> tuple[list, list | None] | tuple[list, list | None, dict]:
        parts = [
            f'原始文本：{self._get_text(obj)}',
            self.config.special_tokens['part_separator'],
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
                        self.config.special_tokens['entity_prefix'],
                        f'{entities[text_split]}：{text_split}',
                        self.config.special_tokens['entity_postfix'],
                    ])
                else:
                    label.append(text_split)
        return self._prepend_start_token(parts), label


class CUGENERDataset(BaseNERDataset):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_entities(obj) -> list[dict]:
        return obj['entities']

    @staticmethod
    def _has_label(obj) -> bool:
        return 'entities' in obj


class CMeEEDataset(CUGENERDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_types = {
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
