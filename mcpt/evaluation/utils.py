import json
import time
import pathlib
import contextlib
import dataclasses
import numpy as np
import multiprocessing

from multiprocessing.managers import SyncManager, BaseProxy
from typing import *

import mcpt


def load_dataset(config: Dict[str, Any]) -> \
        Tuple[
            List[Union[np.ndarray, List[np.ndarray]]],
            List[Optional[Union[np.ndarray, List[np.ndarray]]]],
            Optional[List[str]]
        ]:
    dataset = mcpt.datasets.evaluation.load(config)
    items, candidates = dataset.prepare()
    data, label = [], []
    for item in items:
        if config['model_config'].get('use_pinyin', False):
            if isinstance(item['text'], np.ndarray):
                text = np.expand_dims(item['text'], axis=-2)
                pinyin = np.expand_dims(item['pinyin'], axis=-2)
                data.append(np.concatenate((text, pinyin), axis=-2))
            else:
                obj = []
                for i in range(len(item['text'])):
                    text = np.expand_dims(item['text'][i], axis=-2)
                    pinyin = np.expand_dims(item['pinyin'][i], axis=-2)
                    obj.append(np.concatenate((text, pinyin), axis=-2))
                data.append(obj)
        else:
            data.append(item['text'])
        label.append(item['label'])
    return data, label, candidates


def get_output_path(config: Dict[str, Any]) -> str:
    info = {
        'dataset': config['dataset'],
        'model': pathlib.Path(config['model']).stem if config.get('model') is not None else 'None',
        'timestamp': str(int(time.time())),
        'template_id': str(config['template_id']),
        'split': config['split'],
    }
    path = config['output_path_template']
    for k, v in info.items():
        path = path.replace(f'{{{k}}}', v)
    return path


def get_eval_fn(name: str) -> Callable:
    return {
        'generation': mcpt.evaluation.eval_fn.generation,
    }[name]


def get_eval_metric(name: Optional[str]) -> Optional[Callable]:
    if name is None:
        return None
    return {
    }[name]


class EvalOutputReader(Sequence):
    @dataclasses.dataclass
    class _EvalOutputEntry:
        id: int
        input: str
        target: Any
        output: Any

    def __init__(self, path: str, none_as_empty: bool = False):
        self._none_as_empty = none_as_empty
        objs = self._load(path)
        objs.sort(key=lambda obj: obj['id'])
        self._entries = [self._EvalOutputEntry(**obj) for obj in objs]

    def _load(self, path: str) -> List[Dict[str, Any]]:
        objs = mcpt.load_file(path, format='jsonl')
        if self._none_as_empty:
            for obj in objs:
                if obj['target'] is None:
                    obj['target'] = ''
                if obj['output'] is None:
                    obj['output'] = ''
        return objs

    def __getitem__(self, item):
        return self._entries[item]

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)


class BaseCallback:

    def __init__(self, lock: Optional[multiprocessing.Lock] = None):
        self._lock = contextlib.suppress() if lock is None else lock

    def __call__(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class LogResultCallback(BaseCallback):

    def __init__(
            self,
            path: str,
            verbose: int,
            tokenizer: mcpt.Tokenizer,
            use_perplexity: bool = False,
            use_pinyin: bool = False,
            lock: Optional[multiprocessing.Lock] = None,
    ):
        super().__init__(lock)
        self._path = path
        self._verbose = verbose
        self._tokenizer = tokenizer
        self._use_perplexity = use_perplexity
        self._use_pinyin = use_pinyin

    def __call__(
            self,
            x: Union[np.ndarray, List[np.ndarray]],
            y_true: Optional[Union[np.ndarray, List[np.ndarray]]],
            y_pred: Any,
            index: int,
            *args,
            **kwargs,
    ):
        if self._verbose <= 0:
            return
        with self._lock:
            with open(self._path + '.jsonl', 'a') as f:
                if self._use_perplexity:
                    if self._verbose <= 1:
                        f.write(json.dumps({'output': y_pred[0]}, ensure_ascii=False) + '\n')
                    else:
                        f.write(json.dumps({
                            'id': index,
                            'input': [
                                self._tokenizer.convert_ids_to_string(
                                    list(choice[0][0] if self._use_pinyin else choice[0]))
                                for choice in x
                            ],
                            'target': y_true[0] if y_true is not None else None,
                            'output': y_pred[0],
                        }, ensure_ascii=False) + '\n')
                else:
                    if kwargs.get('raw_output', False):
                        output = y_pred
                    else:
                        output = self._tokenizer.convert_ids_to_string(list(y_pred))
                    if self._verbose <= 1:
                        f.write(json.dumps({'output': output}, ensure_ascii=False) + '\n')
                    elif self._verbose >= 2:
                        if y_true is not None:
                            if isinstance(y_true, np.ndarray):
                                y_true = [y_true]
                            y_true = [self._tokenizer.convert_ids_to_string(list(y_true_i)) for y_true_i in y_true]
                            if len(y_true) == 1:
                                y_true = y_true[0]
                        f.write(json.dumps({
                            'id': index,
                            'input': self._tokenizer.convert_ids_to_string(list(x[0][0] if self._use_pinyin else x[0])),
                            'target': y_true,
                            'output': output,
                        }, ensure_ascii=False) + '\n')


class ProgressBarCallback(BaseCallback):

    def __init__(self, total: Optional[int] = None, lock: Optional[multiprocessing.Lock] = None, **kwargs):
        super().__init__(lock)
        self._bar = mcpt.tqdm(total=total, **kwargs)

    def __call__(self, *args, **kwargs):
        with self._lock:
            self._bar.update(1)

    def close(self, *args, **kwargs):
        self._bar.close()


class CallbackManager(SyncManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register('ProgressBarCallback', ProgressBarCallback, CallbackProxy)
        self.register('LogResultCallback', LogResultCallback, CallbackProxy)


class CallbackProxy(BaseProxy):
    _exposed_ = ('__call__', 'close')

    def __call__(self, *args, **kwargs):
        self._callmethod('__call__', args, kwargs)

    def close(self, *args, **kwargs):
        self._callmethod('close', args, kwargs)
