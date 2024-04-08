import fire
import torch
import pathlib
import collections
import tensorflow as tf

import mcpt_tf
import linglong


def set_weight(
        torch_key: str,
        tf_key: str,
        torch_weights: dict[str, torch.Tensor],
        tf_weights: dict[str, tf.Variable],
):
    tf_weight = tf_weights[tf_key]
    torch_weight = torch_weights[torch_key]
    if list(torch_weight.size()) != list(tf_weight.shape):
        torch_weight = torch.unsqueeze(torch_weight, dim=0)
    assert list(torch_weight.size()) == list(tf_weight.shape)
    print(f'{torch_key}: {list(torch_weight.size())} -> {tf_key}: {list(tf_weight.shape)}')
    tf_weight.assign(tf.Variable(torch_weight.numpy(), dtype=tf.float32))


def tf_idx(idx: int) -> str:
    return f'_{idx}' if idx > 0 else ''


def main(
        model_config: str,
        torch_model_path: str,
        tf_model_path: str | None = None,
):
    tf_model = mcpt_tf.Model.from_config_(model_config)
    torch_model = linglong.Model.from_config(model_config, load_model=torch_model_path, device='cpu')

    tf_weights = collections.OrderedDict()
    for tf_weight in tf_model.weights:
        tf_weights[tf_weight.name] = tf_weight
    print(f'Loaded {len(tf_weights)} weights from the TensorFlow model: '
          f'{linglong.prettify(list(tf_weights.keys()))}')

    torch_weights = torch_model.state_dict()
    print(f'Loaded {len(torch_weights)} weights from the PyTorch model: '
          f'{linglong.prettify(list(torch_weights.keys()))}')

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
    tf_model.set_weights(tf.keras.backend.batch_get_value(tf_weights.values()))
    if tf_model_path is None:
        tf_model_path = pathlib.Path(torch_model_path).with_suffix('.h5')
    tf_model.save_weights(tf_model_path)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
