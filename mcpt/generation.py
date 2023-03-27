import warnings
import numpy as np

from typing import *

import mcpt


def convert_prompt_to_ids(
        prompt: str,
        tokenizer: mcpt.tokenization.Tokenizer,
        pinyin_tokenizer: mcpt.tokenization.PinyinTokenizer,
        special_tokens: Dict[str, str],
        use_pinyin: bool = False,
) -> Union[List[int], List[List[int]]]:
    prompt_parts = prompt.split(special_tokens['part-separator'])
    prompt_text_ids = tokenizer.convert_tokens_to_ids([special_tokens['start-token']])
    for prompt_part in prompt_parts:
        prompt_text_ids.extend(tokenizer.convert_string_to_ids(prompt_part))
        prompt_text_ids.append(tokenizer.convert_tokens_to_ids(special_tokens['part-separator']))
    prompt_text_ids = prompt_text_ids[:-1]

    if use_pinyin:
        prompt_pinyin_ids = pinyin_tokenizer.convert_tokens_to_ids([special_tokens['start-token']])
        for prompt_part in prompt_parts:
            prompt_pinyin_ids.extend(pinyin_tokenizer.convert_string_to_ids(prompt_part))
            prompt_pinyin_ids.append(
                pinyin_tokenizer.convert_tokens_to_ids(special_tokens['part-separator'])
            )
        prompt_pinyin_ids = prompt_pinyin_ids[:-1]
        if len(prompt_text_ids) != len(prompt_pinyin_ids):
            warnings.warn(f'`text` has size {len(prompt_text_ids)} and `pinyin` has size {len(prompt_pinyin_ids)}.'
                          f' (most likely due to omitted control characters).'
                          f'The pinyin information is discarded.')
            prompt_pinyin_ids = [0] * len(prompt_text_ids)
        prompt_ids = [prompt_text_ids, prompt_pinyin_ids]
    else:
        prompt_ids = prompt_text_ids

    return prompt_ids


def process_samples(
        samples,
        prompt_ids: List[int],
        end_id: int,
        tokenizer: mcpt.tokenization.Tokenizer,
):
    # Exclude the start token.
    samples = np.asarray(samples.to('cpu'))[:, 1:]
    prompt_ids = prompt_ids[1:]
    for sample in samples:
        if end_id in sample:
            sample = sample[:sample.tolist().index(end_id)]
        sample = sample[len(prompt_ids):]
        text_prompt = tokenizer.convert_ids_to_string(prompt_ids)
        text_generated = tokenizer.convert_ids_to_string(sample)
        yield text_prompt, text_generated
