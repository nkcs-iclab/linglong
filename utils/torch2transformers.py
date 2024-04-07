import fire
import torch
import pathlib

import linglong


def set_weight(
        torch_key: str,
        transformers_key: str,
        torch_weights: dict[str, torch.Tensor],
        transformers_weights: dict[str, torch.Tensor],
):
    transformers_weight = transformers_weights[transformers_key]
    torch_weight = torch_weights[torch_key]
    assert list(torch_weight.size()) == list(transformers_weight.size())
    print(f'{torch_key}: {list(torch_weight.size())} -> {transformers_key}: {list(transformers_weight.shape)}')
    transformers_weights[transformers_key] = torch_weight


def main(
        model_config: str,
        torch_model_path: str,
        transformers_model_path: str | None = None,
        vocab: str = '../common/vocab/char-13312.txt',
):
    transformers_model = linglong.LingLongModel(linglong.LingLongConfig())
    torch_model = linglong.Model.from_config(model_config, load_model=torch_model_path, device='cpu')
    transformers_weights = transformers_model.state_dict()
    print(f'Loaded {len(transformers_weights)} weights from the Transformers model: '
          f'{linglong.pprint(list(transformers_weights.keys()), export=True)}')
    torch_weights = torch_model.state_dict()
    print(f'Loaded {len(torch_weights)} weights from the PyTorch model: '
          f'{linglong.pprint(list(torch_weights.keys()), export=True)}')

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
    transformers_model.load_state_dict(transformers_weights)
    if transformers_model_path is None:
        transformers_model_path = pathlib.Path(torch_model_path).with_suffix('')
    transformers_model.save_pretrained(transformers_model_path)

    tokenizer = linglong.Tokenizer(vocab)
    tokenizer.save_pretrained(transformers_model_path)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
