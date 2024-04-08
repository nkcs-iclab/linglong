import sys
import json
import yaml
import torch
import colorama
import warnings
import deepmerge
import contextlib

from typing import Callable
from tqdm import tqdm as tqdm_tqdm
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


def _version_str_to_tuple(version_str: str) -> tuple[int, ...]:
    version_str = version_str.split('+')[0]
    return tuple(map(int, version_str.split('.')))


def _check_version(
        ver: str | tuple[int, ...],
        min_ver: str | None = None,
        max_ver: str | None = None,
) -> bool:
    if isinstance(ver, str):
        ver = _version_str_to_tuple(ver)
    status = True
    if min_ver is not None:
        status = status and ver >= _version_str_to_tuple(min_ver)
    if max_ver is not None:
        status = status and ver <= _version_str_to_tuple(max_ver)
    return status


def torch_version(min_ver: str | None = None, max_ver: str | None = None) -> bool:
    return _check_version(_version_str_to_tuple(torch.__version__), min_ver, max_ver)


def python_version(min_ver: str | None = None, max_ver: str | None = None) -> bool:
    return _check_version(sys.version_info, min_ver, max_ver)


def init(window_color_fix: bool = True):
    if window_color_fix:
        # noinspection PyUnresolvedReferences
        colorama.just_fix_windows_console()


def tqdm(*args, **kwargs) -> tqdm_tqdm:
    is_main_process = kwargs.pop('is_main_process', True)
    if is_main_process:
        return tqdm_tqdm(*args, ncols=80, file=sys.stdout, ascii='.=', **kwargs)
    return args[0] if len(args) > 0 else Noop()


def trange(*args, **kwargs) -> tqdm_tqdm:
    return tqdm(range(*args), **kwargs)


def text(msg: str, style: str | None = None) -> str:
    return (_color_theme[style] + str(msg) + colorama.Fore.RESET) if style is not None else str(msg)


# noinspection PyShadowingBuiltins
def load_config(path: str, key: str | None = None, format: str | None = None) -> dict:
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


def merge_configs(*configs: dict) -> dict:
    merged = {}
    for config in configs:
        merged = deepmerge.always_merger.merge(merged, config)
    return merged


# noinspection PyShadowingBuiltins
def load_file(path: str, format: str | None = None) -> list | dict:
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


def prettify(
        d: list | dict,
        indent: int | str | None = 2,
        ensure_ascii: bool = False,
        **kwargs,
) -> str | None:
    return json.dumps(d, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def print_training_records(records, tokenizer: 'linglong.Tokenizer', print_fn: Callable = print):
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
        example = {'data': tokenizer.decode(input_ids[i])}
        if pinyin_input_ids is not None:
            example['pinyin_input_ids'] = str(pinyin_input_ids[i].numpy().tolist())
        example['attention_mask'] = str(attention_mask[i].numpy().tolist())
        example['label_ids'] = tokenizer.decode(label_ids[i])
        example['data[data != 0].shape'] = input_ids[i][input_ids[i] != 0].shape[0]
        example['attention_mask[attention_mask != 0].shape'] = attention_mask[i][attention_mask[i] != 0].shape[0]
        example['label_ids[label_ids != -100].shape'] = label_ids[i][label_ids[i] != -100].shape[0]
        output['examples'].append(example)
    print_fn(prettify(output))


def print_trainable_parameters(model, is_main_process: bool = True, print_fn: Callable = print):
    if not is_main_process:
        return
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_el = param.ds_numel if hasattr(param, 'ds_tensor') else param.numel()
        all_param += num_el
        if param.requires_grad:
            trainable_params += num_el
    print_fn(prettify({
        'trainable_params': trainable_params,
        'all_params': all_param,
        'trainable_ratio': f'{100 * trainable_params / all_param:.3f}%',
    }))


@contextlib.contextmanager
def running(message: str, spinner: bool = True, is_main_process: bool = True, print_fn: Callable = print, **kwargs):
    if spinner:
        if is_main_process:
            with yaspin(Spinners.line, text=text(message, style=INFO), **kwargs) as spinner:
                yield spinner
            spinner.ok(text('(OK!)', style=SUCCESS))
            return
        yield Writable(print_fn=lambda *_: None)
    else:
        if is_main_process:
            print_fn(text(message, style=INFO))
        yield Writable()
