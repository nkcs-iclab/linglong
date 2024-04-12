import re
import abc
import numpy as np

import linglong


def report_accuracy(correct: int, total: int) -> dict:
    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total,
    }


class MetricBase(metaclass=abc.ABCMeta):

    def __init__(self, tokenizer: linglong.Tokenizer):
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def call(self, examples, **kwargs) -> dict | None:
        pass

    def __call__(self, examples, **kwargs) -> dict | None:
        return self.call(examples, **kwargs)


class DatasetMetric(MetricBase, metaclass=abc.ABCMeta):
    pass


class TaskTypeMetric(MetricBase, metaclass=abc.ABCMeta):
    pass


class Math23kDatasetMetric(DatasetMetric):

    @staticmethod
    def _remove_mixed_number(line: str) -> str:
        pattern = re.compile(r'\d\(')
        while re_result := pattern.search(line):
            span = re_result.span()
            end = span[1] - 1
            line = f'{line[:end]}+{line[end:]}'
        return line

    @staticmethod
    def _remove_percentage(line: str) -> str:
        pattern = re.compile(r'\d+\.*\d*%')
        while re_result := pattern.search(line):
            span = re_result.span()
            line = f'{line[:span[0]]}({line[span[0]:span[1] - 1]}/100){line[span[1]:]}'
        return line

    def call(self, examples, **kwargs) -> dict | None:
        correct = 0
        for example in examples:
            if 'label_ids' not in example:
                return None
            line = None
            try:
                prediction = self.tokenizer.decode(example['generated_ids'])
                label = self.tokenizer.decode(example['label_ids'])
                line = f'{prediction}={label}'
                line = line.replace('[', '(').replace(']', ')')
                line = line.replace('^', '**')

                # Special cases.
                line = line.replace('千米/小时', '')

                if re.compile(r'\d\(').search(line):
                    line = self._remove_mixed_number(line)
                if '%' in line:
                    line = self._remove_percentage(line)
                prediction, label = line.split('=', maxsplit=1)
                if np.isclose(float(eval(prediction)), float(eval(label))):
                    correct += 1
            except Exception as e:
                print(f'ERROR: {e} at example prediction=label: {line}')
        return report_accuracy(correct, len(examples))
