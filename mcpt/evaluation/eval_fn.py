import numpy as np

from typing import *

import mcpt


def generation(
        x: List[np.ndarray],
        model: mcpt.Model,
        callbacks: Optional[Callable] = None,
        **kwargs,
) -> List[np.ndarray]:
    generation_config = {
        'max_length': kwargs['config'].get('extra_config', {}).get('max_length', 256),
        'temperature': kwargs['config'].get('extra_config', {}).get('temperature', 1.0),
        'top_k': kwargs['config'].get('extra_config', {}).get('top_k', 1),
        'top_p': kwargs['config'].get('extra_config', {}).get('top_p', 1.0),
        'batch_size': kwargs['config'].get('extra_config', {}).get('batch_size', 1),
    }
    if kwargs['candidates'] is not None and kwargs['special_token_ids']['end_token'] not in kwargs['candidates']:
        kwargs['candidates'].append(kwargs['special_token_ids']['end_token'])
    y_pred = []
    sampler = mcpt.generation.Sampler(
        model=model,
        end_id=kwargs['special_token_ids']['end_token'],
        device=kwargs['device'],
        pinyin_tokenizer=kwargs['pinyin_tokenizer'],
        tokenizer=kwargs['tokenizer'],
        verbose=0,
    )
    use_pinyin = model.config.get('use_pinyin', False)
    for idx, (data, label) in enumerate(zip(x, kwargs['y_true'])):
        samples = sampler.batch_sample(data[0], generation_config, kwargs['candidates'])
        items = []
        for pred in np.asarray(samples.to('cpu'))[:, len(data[0][0] if use_pinyin else data[0]):]:
            if kwargs['special_token_ids']['end_token'] in pred:
                pred = pred[:pred.tolist().index(kwargs['special_token_ids']['end_token'])]
            items.append(pred)
            items.append(kwargs['tokenizer'].convert_string_to_ids(
                kwargs['config'].get('extra_config', {}).get('merge_tokens', '')
            ))
        if merge_method := kwargs['config'].get('extra_config', {}).get('merge_method', 'concat') == 'concat':
            pred = np.concatenate(items[:-1])
        else:
            raise ValueError(f'Invalid merge method: {merge_method}')
        y_pred.append(pred)
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            if callback is not None:
                callback(data, label, pred, kwargs.get('offset', 0) + idx)
    return y_pred
