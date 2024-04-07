import sys
import json
import yaml
import torch
import colorama
import warnings
import deepmerge
import contextlib

from tqdm import tqdm as tqdm_tqdm
from typing import *
from yaspin import yaspin
from yaspin.spinners import Spinners

import linglong

_color_theme = {
    'info': colorama.Fore.CYAN,
    'success': colorama.Fore.LIGHTGREEN_EX,
    'error': colorama.Fore.LIGHTRED_EX,
    'warning': colorama.Fore.YELLOW,
    'structure': colorama.Fore.LIGHTWHITE_EX,
}

INFO = 'info'
SUCCESS = 'success'
WARNING = 'warning'
ERROR = 'error'
STRUCTURE = 'structure'


class Comm:

    def __init__(self, size: int = 1, rank: int = 0, local_rank: int = 0):
        self._size = size
        self._rank = rank
        self._local_rank = local_rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def local_rank(self) -> int:
        return self._local_rank


class Writable:

    def __init__(self, print_fn: Callable = print):
        self._print_fn = print_fn

    def write(self, *args, **kwargs):
        self._print_fn(*args, **kwargs)


class Noop:

    def noop(self, *_, **__):
        return self

    def __getattr__(self, item):
        return self.noop


def _version_str_to_tuple(version_str: str) -> Tuple[int, ...]:
    version_str = version_str.split('+')[0]
    return tuple(map(int, version_str.split('.')))


def _check_version(
        ver: Union[str, Tuple[int, ...]],
        min_ver: Optional[str] = None,
        max_ver: Optional[str] = None,
) -> bool:
    if isinstance(ver, str):
        ver = _version_str_to_tuple(ver)
    status = True
    if min_ver is not None:
        status = status and ver >= _version_str_to_tuple(min_ver)
    if max_ver is not None:
        status = status and ver <= _version_str_to_tuple(max_ver)
    return status


def torch_version(min_ver: Optional[str] = None, max_ver: Optional[str] = None) -> bool:
    return _check_version(_version_str_to_tuple(torch.__version__), min_ver, max_ver)


def python_version(min_ver: Optional[str] = None, max_ver: Optional[str] = None) -> bool:
    return _check_version(sys.version_info, min_ver, max_ver)


def init(window_color_fix: bool = True):
    if window_color_fix:
        # noinspection PyUnresolvedReferences
        colorama.just_fix_windows_console()


def tqdm(*args, **kwargs) -> tqdm_tqdm:
    comm: Comm | None = kwargs.pop('comm', None)
    if comm is not None and comm.rank() != 0:
        return args[0] if len(args) > 0 else Noop()
    return tqdm_tqdm(*args, ncols=80, file=sys.stdout, ascii='.=', **kwargs)


def trange(*args, **kwargs) -> tqdm_tqdm:
    return tqdm(range(*args), **kwargs)


def text(msg: str, style: Optional[str] = None) -> str:
    return (_color_theme[style] + str(msg) + colorama.Fore.RESET) if style is not None else str(msg)


# noinspection PyShadowingBuiltins
def load_config(path: str, key: Optional[str] = None, format: Optional[str] = None) -> Dict[str, Any]:
    if format is None:
        format = path.split('.')[-1]
    format = format.lower()
    with open(path, 'r') as f:
        if format == 'json':
            config = json.load(f)
        elif format == 'yaml' or format == 'yml':
            config = yaml.safe_load(f)
        else:
            warnings.warn(f'Cannot determine the config format of {path} from the extension. '
                          'Assuming it is a YAML file.')
            config = yaml.safe_load(f)
    if key is not None:
        config = config[key]
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    merged = {}
    for config in configs:
        merged = deepmerge.always_merger.merge(merged, config)
    return merged


# noinspection PyShadowingBuiltins
def load_file(path: str, format: Optional[str] = None) -> Union[List, Dict]:
    if format is None:
        format = path.split('.')[-1]
    format = format.lower()
    with open(path, 'r', encoding='utf-8') as f:
        if format == 'txt':
            data = f.read().splitlines()
        elif format == 'json':
            data = json.load(f)
        elif format == 'jsonl':
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f'Unknown file format {format}.')
    return data


def pprint(
        d: Union[Dict, List],
        indent: Optional[Union[int, str]] = 2,
        ensure_ascii: bool = False,
        export: bool = False,
        **kwargs,
) -> Optional[str]:
    formatted = json.dumps(d, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
    if export:
        return formatted
    print(formatted)


def print_training_records(records, tokenizer: 'linglong.Tokenizer'):
    input_ids = records['input_ids']
    pinyin_input_ids = records.get('pinyin_input_ids', None)
    attention_mask = records['attention_mask']
    label_ids = records['label_ids']
    output = {
        'shape': {
            'input_ids': input_ids.shape,
            'pinyin_input_ids': pinyin_input_ids.shape if pinyin_input_ids is not None else None,
            'attention_mask': attention_mask.shape,
            'label_ids': label_ids.shape,
        },
        'examples': [],
    }
    for i in range(len(input_ids)):
        example = {}
        example['data'] = tokenizer.decode(input_ids[i])
        if pinyin_input_ids is not None:
            example['pinyin_input_ids'] = str(pinyin_input_ids[i].numpy().tolist())
        example['attention_mask'] = str(attention_mask[i].numpy().tolist())
        example['label_ids'] = tokenizer.decode(label_ids[i])
        example['data[data != 0].shape'] = input_ids[i][input_ids[i] != 0].shape[0]
        example['attention_mask[attention_mask != 0].shape'] = attention_mask[i][attention_mask[i] != 0].shape[0]
        example['label_ids[label_ids != -100].shape'] = label_ids[i][label_ids[i] != -100].shape[0]
        output['examples'].append(example)
    pprint(output)


def print_trainable_parameters(model, comm=None, print_fn: Callable = print):
    if comm is None or comm.rank() <= 0:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            numel = param.ds_numel if hasattr(param, 'ds_tensor') else param.numel()
            all_param += numel
            if param.requires_grad:
                trainable_params += numel
        print_fn(
            f'trainable params: {trainable_params} || '
            f'all params: {all_param} || '
            f'trainable%: {100 * trainable_params / all_param}',
        )


@contextlib.contextmanager
def running(text_: str, spinner: bool = True, comm=None, **kwargs):
    if spinner:
        if comm is not None:
            if comm.rank() != 0:
                yield Writable(print_fn=lambda *_: None)
                return
        with yaspin(Spinners.line, text=text(text_, style=INFO), **kwargs) as spinner:
            yield spinner
        spinner.ok(text('(OK!)', style=SUCCESS))
    else:
        if comm is None or comm.rank() <= 0:
            print(text(text_, style=INFO))
        yield Writable()
