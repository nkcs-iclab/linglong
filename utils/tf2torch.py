import fire
import torch
import pathlib
import tensorflow as tf

from typing import *

import mcpt_tf
import mcpt


def set_weight(torch_key: str, tf_key: str, torch_weights: Dict[str, torch.Tensor], tf_weights: Dict[str, tf.Variable]):
    tf_weight = tf_weights[tf_key].numpy()
    if list(torch_weights[torch_key].size()) != list(tf_weight.shape):
        tf_weight = tf_weight[0]
    assert list(torch_weights[torch_key].size()) == list(tf_weight.shape)
    print(f'{torch_key}: {torch_weights[torch_key].size()} <- {tf_key}: {tf_weight.shape}')
    torch_weights[torch_key] = torch.from_numpy(tf_weight)


def tf_idx(idx: int) -> str:
    return f'_{idx}' if idx > 0 else ''


def main(
        model_config: str,
        tf_model_path: str,
        torch_model_path: Optional[str] = None,
):
    tf_model = mcpt_tf.Model.from_config_(model_config, load_model=tf_model_path)
    torch_model = mcpt.Model.from_config(model_config)

    tf_weights = tf_model.weights
    tf_weights = {w.name: w for w in tf_weights}
    print(f'Loaded {len(tf_weights)} weights from the TensorFlow model: '
          f'{mcpt.pprint(list(tf_weights.keys()), export=True)}')

    torch_weights = torch_model.state_dict()
    print(f'Loaded {len(torch_weights)} weights from the PyTorch model: '
          f'{mcpt.pprint(list(torch_weights.keys()), export=True)}')

    set_weight(
        'transformer.wte.weight',
        'mcpt_model/embedding/embedding_1/embeddings:0',
        torch_weights,
        tf_weights,
    )
    set_weight(
        'transformer.wpe.weight',
        'Variable:0',
        torch_weights,
        tf_weights,
    )

    ln_idx = 0
    attn_idx = 0
    mlp_idx = 0
    conv1d_idx = 0
    for i in range(torch_model.config['n_layer']):
        set_weight(
            f'transformer.blocks.{i}.ln_1.weight',
            f'mcpt_model/block{tf_idx(i)}/layer_normalization{tf_idx(ln_idx)}/gamma:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.ln_1.bias',
            f'mcpt_model/block{tf_idx(i)}/layer_normalization{tf_idx(ln_idx)}/beta:0',
            torch_weights,
            tf_weights,
        )
        ln_idx += 1

        set_weight(
            f'transformer.blocks.{i}.attn.c_attn.w',
            f'mcpt_model/block{tf_idx(i)}/attention{tf_idx(attn_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_w:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_attn.b',
            f'mcpt_model/block{tf_idx(i)}/attention{tf_idx(attn_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_b:0',
            torch_weights,
            tf_weights,
        )
        conv1d_idx += 1

        set_weight(
            f'transformer.blocks.{i}.attn.c_proj.w',
            f'mcpt_model/block{tf_idx(i)}/attention{tf_idx(attn_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_w:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.attn.c_proj.b',
            f'mcpt_model/block{tf_idx(i)}/attention{tf_idx(attn_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_b:0',
            torch_weights,
            tf_weights,
        )
        attn_idx += 1
        conv1d_idx += 1

        set_weight(
            f'transformer.blocks.{i}.ln_2.weight',
            f'mcpt_model/block{tf_idx(i)}/layer_normalization{tf_idx(ln_idx)}/gamma:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.ln_2.bias',
            f'mcpt_model/block{tf_idx(i)}/layer_normalization{tf_idx(ln_idx)}/beta:0',
            torch_weights,
            tf_weights,
        )
        ln_idx += 1

        set_weight(
            f'transformer.blocks.{i}.mlp.c_fc.w',
            f'mcpt_model/block{tf_idx(i)}/mlp{tf_idx(mlp_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_w:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_fc.b',
            f'mcpt_model/block{tf_idx(i)}/mlp{tf_idx(mlp_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_b:0',
            torch_weights,
            tf_weights,
        )
        conv1d_idx += 1
        set_weight(
            f'transformer.blocks.{i}.mlp.c_proj.w',
            f'mcpt_model/block{tf_idx(i)}/mlp{tf_idx(mlp_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_w:0',
            torch_weights,
            tf_weights,
        )
        set_weight(
            f'transformer.blocks.{i}.mlp.c_proj.b',
            f'mcpt_model/block{tf_idx(i)}/mlp{tf_idx(mlp_idx)}/conv1d{tf_idx(conv1d_idx)}/conv1d_b:0',
            torch_weights,
            tf_weights,
        )
        mlp_idx += 1
        conv1d_idx += 1

    set_weight(
        'transformer.ln_f.weight',
        f'mcpt_model/layer_normalization{tf_idx(ln_idx)}/gamma:0',
        torch_weights,
        tf_weights,
    )
    set_weight(
        'transformer.ln_f.bias',
        f'mcpt_model/layer_normalization{tf_idx(ln_idx)}/beta:0',
        torch_weights,
        tf_weights,
    )
    torch_model.load_state_dict(torch_weights)
    if torch_model_path is None:
        torch_model_path = pathlib.Path(tf_model_path).with_suffix('.pt')
    torch.save(torch_weights, torch_model_path)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
