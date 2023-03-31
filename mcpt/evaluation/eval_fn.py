import numpy as np

from typing import *

import mcpt


def generation(x: List[np.ndarray], model, callbacks: Optional[Callable] = None, **kwargs) -> List[np.ndarray]:
    generation_config = {
        'length': kwargs['config'].get('extra-config', {}).get('max-length', 256),
        'temperature': kwargs['config'].get('extra-config', {}).get('temperature', 1.0),
        'top_k': kwargs['config'].get('extra-config', {}).get('top-k', 1),
        'top_p': kwargs['config'].get('extra-config', {}).get('top-p', 1.0),
        'batch_size': kwargs['config'].get('extra-config', {}).get('batch-size', 1),
    }
    if kwargs['candidates'] is not None and kwargs['special_token_ids']['end-token'] not in kwargs['candidates']:
        kwargs['candidates'].append(kwargs['special_token_ids']['end-token'])
    y_pred = []
    sampler = mcpt.generation.Sampler(
        model_config=kwargs['model_config'],
        model=model,
        end_id=kwargs['special_token_ids']['end-token'],
        device=kwargs['device'],
        pinyin_tokenizer=kwargs['pinyin_tokenizer'],
        tokenizer=kwargs['tokenizer'],
        use_pinyin=kwargs['config']['use-pinyin'],
    )
    for idx, (data, label) in enumerate(zip(x, kwargs['y_true'])):
        samples = sampler.batch_sample(data[0], generation_config, kwargs['candidates'])
        items = []
        for pred in np.asarray(samples.to('cpu'))[:, len(data[0][0] if kwargs['config']['use-pinyin'] else data[0]):]:
            if kwargs['special_token_ids']['end-token'] in pred:
                pred = pred[:pred.tolist().index(kwargs['special_token_ids']['end-token'])]
            items.append(pred)
            items.append(kwargs['tokenizer'].convert_string_to_ids(
                kwargs['config'].get('extra-config', {}).get('merge_tokens', '')
            ))
        if merge_method := kwargs['config'].get('extra-config', {}).get('merge_method', 'concat') == 'concat':
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
