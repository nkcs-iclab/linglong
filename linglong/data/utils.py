import torch

from typing import Callable
from torch.utils.data import Dataset

import linglong


def pad_sequence(
        sequences,
        batch_first: bool = False,
        padding_value: float = 0.0,
        padding_side: str = 'right',
):
    if padding_side == 'right':
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    elif padding_side == 'left':
        sequences = list(map(lambda s: s.flip(0), sequences))
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim + batch_first)
    else:
        raise ValueError(f'`padding_side` should be either "right" or "left", but got {padding_side}')
    return padded_sequence


def padded_batch(batch):
    output = {}
    keys = batch[0].keys()
    for k in keys:
        if k == 'label_ids':
            padding_side = 'right'
            padding_value = -100
        elif k == 'id':
            output[k] = torch.tensor([x[k] for x in batch])
            continue
        else:
            padding_side = 'left'
            padding_value = 0
        output[k] = pad_sequence(
            [x[k] for x in batch],
            batch_first=True,
            padding_value=padding_value,
            padding_side=padding_side,
        )
    return output


def print_model_inputs(
        batch,
        tokenizer: linglong.LingLongTokenizer | linglong.LingLongTokenizerFast,
        print_fn: Callable = print,
):
    input_ids = batch['input_ids']
    pinyin_input_ids = batch.get('pinyin_input_ids', None)
    attention_mask = batch['attention_mask']
    label_ids = batch['label_ids']
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
        example = {
            'input': tokenizer.decode(input_ids[i]),
            'input_ids': str(input_ids[i].numpy().tolist()),
        }
        if pinyin_input_ids is not None:
            example['pinyin_input_ids'] = str(pinyin_input_ids[i].numpy().tolist())
        example.update({
            'attention_mask': str(attention_mask[i].numpy().tolist()),
            'label': tokenizer.decode(label_ids[i]),
            'label_ids': str(label_ids[i].numpy().tolist()),
            'data[data != 0].shape': input_ids[i][input_ids[i] != 0].shape[0],
            'attention_mask[attention_mask != 0].shape': attention_mask[i][attention_mask[i] != 0].shape[0],
            'label_ids[label_ids != -100].shape': label_ids[i][label_ids[i] != -100].shape[0],
        })
        output['examples'].append(example)
    print_fn(linglong.prettify(output))


class DictDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        label_ids = torch.tensor(item['label_ids'], dtype=torch.long) if 'label_ids' in item else None
        if 'attention_mask' in item:
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        else:
            attention_mask = torch.ones_like(input_ids)
        if 'pinyin_input_ids' in item:
            pinyin_input_ids = torch.tensor(item['pinyin_input_ids'], dtype=torch.long)
            return {
                'id': idx,
                'input_ids': input_ids,
                'pinyin_input_ids': pinyin_input_ids,
                'attention_mask': attention_mask,
                **({'label_ids': label_ids} if label_ids is not None else {}),
            }
        return {
            'id': idx,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            **({'label_ids': label_ids} if label_ids is not None else {}),
        }
