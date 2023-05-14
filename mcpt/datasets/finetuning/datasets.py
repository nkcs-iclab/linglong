from typing import *

from mcpt.datasets.finetuning.base import BaseDataset


class Math23KDataset(BaseDataset):

    def _template_0(self, obj) -> List[Union[str, List[str], Dict[str, List[str]]]]:
        return [
            f'问题：{obj["text"]}',
            [self._special_tokens['part_separator']],
            f'答案：{obj["equation"][2:]}',
        ]
