import fire
import numpy as np

from typing import *

import mcpt
import mcpt.evaluation


def main(
        dataset: str,
        input_path: str,
        cache_path: str,
        dataset_config: str = '../evaluation/configs/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: Optional[str] = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        special_tokens: Optional[Dict[str, str]] = None,
        slicer: Optional[str] = '0:3',
):
    with mcpt.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '[MASK]',
            'end_token': '[CLS]',
            'part_separator': '[unused1]',
            'segment_separator': '[unused2]',
            **(special_tokens or {}),
        }
        config = mcpt.merge_configs({
            'dataset': dataset,
            'dataset_config_path': dataset_config,
            'input_path': input_path,
            'cache_path': cache_path,
            'vocab': vocab,
            'pinyin_vocab': pinyin_vocab,
            'use_cache': use_cache,
            'special_tokens': special_tokens,
        }, mcpt.load_config(dataset_config, key=dataset))
        config['model_config'] = mcpt.load_config(config['model']['config'])
        tokenizer = mcpt.Tokenizer(vocab)
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running(f'Loading {dataset} dataset', spinner=use_cache):
        x, y_true, candidates = mcpt.evaluation.load_dataset(config)
        if slicer is not None:
            slicer = slice(*(int(x) for x in slicer.split(':')))
            x, y_true = x[slicer], y_true[slicer]

    print(mcpt.text('Examples:', style=mcpt.INFO))
    output: Dict[str, Any] = {
        'example_count': len(x),
        'examples': [],
    }
    for i in range(len(x)):
        example: Dict[str, Optional[Union[str, List[int]]]] = {}
        if isinstance(x[i], np.ndarray):
            x[i] = [x[i]]
        x_ids = [str(_.tolist()) for _ in x[i]]
        x_str = [
            tokenizer.convert_ids_to_string(list(_[0][0] if config['model_config'].get('use_pinyin', False) else _[0]))
            for _ in x[i]]
        example['x'] = x_ids if len(x_ids) > 1 else x_ids[0]
        example['x_str'] = x_str if len(x_str) > 1 else x_str[0]
        if y_true[i] is not None:
            if isinstance(y_true[i], np.ndarray):
                y_true[i] = [y_true[i]]
            y_true_ids = [str(_.tolist()) for _ in y_true[i]]
            example['y_true'] = y_true_ids if len(y_true_ids) > 1 else y_true_ids[0]
            if not config.get('use_perplexity', False):
                y_true_str = [tokenizer.convert_ids_to_string(list(_)) for _ in y_true[i]]
                example['y_true_str'] = y_true_str if len(y_true_str) > 1 else y_true_str[0]
        else:
            example['y_true'] = None
        output['examples'].append(example)
    output['candidates'] = candidates
    mcpt.pprint(output)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
