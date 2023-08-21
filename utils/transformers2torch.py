import fire
import torch
import pathlib

from typing import *

import mcpt


def set_weight(
        torch_key: str,
        transformers_key: str,
        torch_weights: Dict[str, torch.Tensor],
        transformers_weights: Dict[str, torch.Tensor],
):
    if 'v_head.weight' in transformers_weights:
        if transformers_key != 'v_head.weight':
            transformers_key = 'rwtranrsformer.' + transformers_key
    else:
        if transformers_key != 'lm_head.weight':
            transformers_key = 'transformer.' + transformers_key
    transformers_weight = transformers_weights[transformers_key]
    torch_weight = torch_weights[torch_key]
    if list(torch_weight.size()) != list(transformers_weight.size()):
        raise ValueError(
            f'Inconsistent weight shape: {torch_key}: {list(torch_weight.size())} vs. '
            f'{transformers_key}: {list(transformers_weight.size())}'
        )
    print(f'{transformers_key}: {list(transformers_weight.size())} -> {torch_key}: {list(torch_weight.size())}')
    torch_weights[torch_key] = transformers_weight


def main(
        model_config: str,
        transformers_model_path: str,
        torch_model_path: Optional[str] = None,
):
    transformers_weights = torch.load(transformers_model_path, map_location='cpu')
    print(f'Loaded {len(transformers_weights)} weights from the Transformers model: '
          f'{mcpt.pprint(list(transformers_weights.keys()), export=True)}')

    if 'v_head.weight' in transformers_weights:
        torch_model = mcpt.RewardModel.from_config(model_config)
    else:
        torch_model = mcpt.Model.from_config(model_config)
    torch_weights = torch_model.state_dict()
    print(f'Loaded {len(torch_weights)} weights from the PyTorch model: '
          f'{mcpt.pprint(list(torch_weights.keys()), export=True)}')

    set_weight(
        'transformer.wte.weight',
        'wte.weight',
        torch_weights,
        transformers_weights,
    )
    set_weight(
        'transformer.wpe.weight',
        'wpe.weight',
        torch_weights,
        transformers_weights,
    )
    for i in range(torch_model.config['n_layer']):
        set_weight(
            f'transformer.blocks.{i}.ln_1.weight',
            f'h.{i}.ln_1.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.ln_1.bias',
            f'h.{i}.ln_1.bias',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_attn.w',
            f'h.{i}.attn.c_attn.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_attn.b',
            f'h.{i}.attn.c_attn.bias',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_proj.w',
            f'h.{i}.attn.c_proj.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_proj.b',
            f'h.{i}.attn.c_proj.bias',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.ln_2.weight',
            f'h.{i}.ln_2.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.ln_2.bias',
            f'h.{i}.ln_2.bias',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_fc.w',
            f'h.{i}.mlp.c_fc.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_fc.b',
            f'h.{i}.mlp.c_fc.bias',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_proj.w',
            f'h.{i}.mlp.c_proj.weight',
            torch_weights,
            transformers_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_proj.b',
            f'h.{i}.mlp.c_proj.bias',
            torch_weights,
            transformers_weights,
        )

    set_weight(
        'transformer.ln_f.weight',
        f'ln_f.weight',
        torch_weights,
        transformers_weights,
    )
    set_weight(
        'transformer.ln_f.bias',
        f'ln_f.bias',
        torch_weights,
        transformers_weights,
    )
    if 'v_head.weight' in transformers_weights:
        set_weight(
            'reward_head.weight',
            f'v_head.weight',
            torch_weights,
            transformers_weights,
        )
    torch_model.load_state_dict(torch_weights)
    if torch_model_path is None:
        torch_model_path = pathlib.Path(transformers_model_path).with_suffix('.pt')
    torch.save(torch_weights, torch_model_path)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
