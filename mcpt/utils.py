import sys
import json
import yaml
import torch
import colorama
import warnings
import mergedeep
import contextlib

from tqdm import tqdm as tqdm_tqdm
from typing import *
from yaspin import yaspin
from yaspin.spinners import Spinners

import mcpt

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


def bind_gpu(hvd=None, gpu_id: Optional[int] = None):
    if hvd is None:
        hvd = mcpt.stubs.Horovod(local_rank=gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())


def tqdm(*args, **kwargs) -> tqdm_tqdm:
    hvd = kwargs.pop('hvd', None)
    if hvd is not None and hvd.rank() != 0:
        return args[0] if len(args) > 0 else mcpt.stubs.Noop()
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
        merged = mergedeep.merge(merged, config)
    return merged


# noinspection PyShadowingBuiltins
def load_file(path: str, format: str = 'txt'):
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


def print_dict(
        d: Dict[str, Any],
        indent: Optional[Union[int, str]] = 2,
        ensure_ascii: bool = False,
        export: bool = False,
        **kwargs,
) -> Optional[str]:
    formatted = json.dumps(d, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
    if export:
        return formatted
    print(formatted)


def print_training_records(records, tokenizer: mcpt.tokenization.Tokenizer):
    data, label, mask = records
    output = {
        'shape': {
            'data': data.shape.as_list(),
            'label': label.shape.as_list(),
            'mask': mask.shape.as_list(),
        },
        'examples': [],
    }
    for i in range(len(data)):
        example = {}
        if len(data.shape) == 3:
            example['data'] = tokenizer.convert_ids_to_string(data[i][0])
            example['pinyin_ids'] = str(data[i][1].numpy().tolist())
        else:
            example['data'] = tokenizer.convert_ids_to_string(data[i])
        example['label'] = tokenizer.convert_ids_to_string(label[i])
        example['label[label != 0].shape'] = label[i][label[i] != 0].shape.as_list()[0]
        example['mask[mask != 0].shape'] = mask[i][mask[i] != 0].shape.as_list()[0]
        output['examples'].append(example)
    print_dict(output)


@contextlib.contextmanager
def running(text_: str, spinner: bool = True, hvd=None, **kwargs):
    if spinner:
        if hvd is not None:
            if hvd.rank() != 0:
                yield mcpt.stubs.Writable(print_fn=lambda *_: None)
                return
        with yaspin(Spinners.line, text=text(text_, style=INFO), **kwargs) as spinner:
            yield spinner
        spinner.ok(text('(OK!)', style=SUCCESS))
    else:
        print(text(text_, style=INFO))
        yield mcpt.stubs.Writable()
