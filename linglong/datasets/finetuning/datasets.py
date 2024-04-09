import re

from linglong.datasets.finetuning.base import FineTuningDatasetBase


class CEPSUM2Dataset(FineTuningDatasetBase):

    def _load_file(self, path: str) -> list | dict:
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

    def _template_0(self, obj) -> list:
        obj_type = {
            'bc': '箱包',
            'cl': '衣服',
            'homea': '家具',
        }[obj['type']]

        features = '；'.join(
            [f'{"".join(k.split())}：{"".join(v.split())}' for k, v in obj['feature'].items()]
        )
        return self._add_start_and_end_tokens([
            (f'商品种类：{obj_type}；特征信息：{features}；商品描述：{"".join(obj["source"].split())}', False),
            (self.config.special_tokens['part_separator'], False),
            ('商品简介：', False),
            (''.join(obj['target'].split()), True),
        ])


class LCSTSDataset(FineTuningDatasetBase):

    def _template_0(self, obj) -> list:
        return self._add_start_and_end_tokens([
            (f'文本：{obj["text"]}', False),
            (self.config.special_tokens['part_separator'], False),
            ('摘要：', False),
            (obj['summary'], True),
        ])


class AdGenDataset(FineTuningDatasetBase):

    def _template_0(self, obj) -> list:
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
        return self._add_start_and_end_tokens([
            (description, False),
            (self.config.special_tokens['part_separator'], False),
            ('商品描述：', False),
            (obj['desc'], True),
        ])


class KBQADataset(FineTuningDatasetBase):

    def _template_0(self, obj) -> list:
        a, relation, _ = obj['triple'].strip().split('|||')
        return self._add_start_and_end_tokens([
            (f'问题：{obj["question"]}', False),
            (self.config.special_tokens['part_separator'], False),
            ('答案：', False),
            (a.strip(), True),
            (self.config.special_tokens['segment_separator'], True),
            (relation.strip(), True),
        ])


class SegmentationDatasetBase(FineTuningDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_segments(obj) -> list[str]:
        raise NotImplementedError

    def _template_0(self, obj) -> list:
        parts = [
            (f'原始文本：{self._get_text(obj)}', False),
            (self.config.special_tokens['part_separator'], False),
            ('分词结果：', False),
        ]
        segments = []
        for segment in self._get_segments(obj):
            segments.append((segment, True))
            segments.append((self.config.special_tokens['segment_separator'], True))
        # Drop the last segment separator.
        parts.extend(segments[:-1])
        return self._add_start_and_end_tokens(parts)


class CUGESegmentationDataset(SegmentationDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_segments(obj) -> list[str]:
        return obj['ans'].split()


class ICWBSegmentationDataset(SegmentationDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        return obj.replace('  ', '')

    @staticmethod
    def _get_segments(obj) -> list[str]:
        return obj.split()


class LCQMCDataset(FineTuningDatasetBase):

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

    def _template_0(self, obj) -> list:
        return self._add_start_and_end_tokens([
            (f'句子一“{obj["sentence1"]}”与句子二“{obj["sentence2"]}”的意思是否相似？', False),
            (self.config.special_tokens['part_separator'], False),
            (['否', '是'][obj['label']], True),
        ])


class Math23KDataset(FineTuningDatasetBase):

    def _template_0(self, obj) -> list:
        return self._add_start_and_end_tokens([
            (f'问题：{obj["text"]}', False),
            (self.config.special_tokens['part_separator'], False),
            ('答案：', False),
            (obj['equation'][2:], True),
        ])


class NERDatasetBase(FineTuningDatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._entity_types = {}

    def entity_type(self, entity_type: str) -> str:
        return self._entity_types[entity_type]

    @staticmethod
    def _get_text(obj) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_entities(obj) -> list[dict]:
        raise NotImplementedError

    def _template_0(self, obj) -> list:
        parts = [
            (f'原始文本：{self._get_text(obj)}', False),
            (self.config.special_tokens['part-separator'], False),
            ('实体：', False),
        ]
        for entity in self._get_entities(obj):
            parts.append((f'{self.entity_type(entity["type"])}：{entity["entity"]}', True))
            parts.append((self.config.special_tokens['segment-separator'], True))
        return self._add_start_and_end_tokens(parts[:-1])

    def _template_1(self, obj) -> list:
        parts = [
            (f'原始文本：{self._get_text(obj)}', False),
            (self.config.special_tokens['part-separator'], False),
            ('实体：', False),
        ]
        entities = {
            entity['entity']: self.entity_type(entity['type'])
            for entity in self._get_entities(obj)
        }
        pattern = rf'({"|".join([re.escape(_) for _ in entities.keys()])})'
        text_splits = re.split(pattern, self._get_text(obj))
        for text_split in text_splits:
            if text_split in entities:
                parts.extend([
                    (self.config.special_tokens['entity-prefix'], True),
                    (f'{entities[text_split]}：{text_split}', True),
                    (self.config.special_tokens['entity-postfix'], True),
                ])
            else:
                parts.append((text_split, True))
        return self._add_start_and_end_tokens(parts)


class CUGENERDataset(NERDatasetBase):

    @staticmethod
    def _get_text(obj) -> str:
        return obj['text']

    @staticmethod
    def _get_entities(obj) -> list[dict]:
        return obj['entities']


class CMeEEDataset(CUGENERDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
