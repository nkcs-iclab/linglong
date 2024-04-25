import abc
import fire
import torch
import pathlib

import numpy as np

import linglong
import mcpt_tf
import tensorflow as tf


def raise_size_mismatch(key: str, src_size: list[int], dst_size: list[int]):
    raise ValueError(f'Invalid shape for {key}: {src_size} != {dst_size}.')


class ModelManager(abc.ABC):
    weight_map = {}
    name = 'default'

    def __init__(self):
        self.model = None
        self.weights = None
        self.n_layer = None

    @abc.abstractmethod
    def get_weight(self, key: str, **info) -> np.ndarray:
        pass

    @abc.abstractmethod
    def set_weight(self, key: str, value: np.ndarray, **info):
        pass

    @abc.abstractmethod
    def save_model(self, save_path: str | None) -> str:
        pass

    def print_loaded_keys(self):
        print(
            f'Loaded {len(self.weights)} weights from the {self.name} model: '
            f'{linglong.prettify(list(self.weights.keys()))}',
        )


class TensorFlowModelManager(ModelManager):
    weight_map = {
        'word_token_embedding': 'mcpt_model/embedding/embedding_1/embeddings:0',
        'position_token_embedding': 'Variable:0',
        'block_layer_norm_1_weight': 'mcpt_model/block{i}/layer_normalization{ln_idx}/gamma:0',
        'block_layer_norm_1_bias': 'mcpt_model/block{i}/layer_normalization{ln_idx}/beta:0',
        'block_attention_weight': 'mcpt_model/block{i}/attention{attn_idx}/conv1d{conv1d_idx}/conv1d_w:0',
        'block_attention_bias': 'mcpt_model/block{i}/attention{attn_idx}/conv1d{conv1d_idx}/conv1d_b:0',
        'block_attention_projection_weight': 'mcpt_model/block{i}/attention{attn_idx}/conv1d{conv1d_idx}/conv1d_w:0',
        'block_attention_projection_bias': 'mcpt_model/block{i}/attention{attn_idx}/conv1d{conv1d_idx}/conv1d_b:0',
        'block_layer_norm_2_weight': 'mcpt_model/block{i}/layer_normalization{ln_idx}/gamma:0',
        'block_layer_norm_2_bias': 'mcpt_model/block{i}/layer_normalization{ln_idx}/beta:0',
        'block_mlp_weight': 'mcpt_model/block{i}/mlp{mlp_idx}/conv1d{conv1d_idx}/conv1d_w:0',
        'block_mlp_bias': 'mcpt_model/block{i}/mlp{mlp_idx}/conv1d{conv1d_idx}/conv1d_b:0',
        'block_mlp_projection_weight': 'mcpt_model/block{i}/mlp{mlp_idx}/conv1d{conv1d_idx}/conv1d_w:0',
        'block_mlp_projection_bias': 'mcpt_model/block{i}/mlp{mlp_idx}/conv1d{conv1d_idx}/conv1d_b:0',
        'layer_norm_weight': 'mcpt_model/layer_normalization{ln_idx}/gamma:0',
        'layer_norm_bias': 'mcpt_model/layer_normalization{ln_idx}/beta:0',
    }
    name = 'TensorFlow'
    layer_idx_multiplier = {
        'ln_idx': 2,
        'attn_idx': 1,
        'conv1d_idx': 4,
        'mlp_idx': 1,
    }
    layer_idx_offset = {
        'word_token_embedding': {},
        'position_token_embedding': {},
        'block_layer_norm_1_weight': {'ln_idx': 0},
        'block_layer_norm_1_bias': {'ln_idx': 0},
        'block_attention_weight': {'attn_idx': 0, 'conv1d_idx': 0},
        'block_attention_bias': {'attn_idx': 0, 'conv1d_idx': 0},
        'block_attention_projection_weight': {'attn_idx': 0, 'conv1d_idx': 1},
        'block_attention_projection_bias': {'attn_idx': 0, 'conv1d_idx': 1},
        'block_layer_norm_2_weight': {'ln_idx': 1},
        'block_layer_norm_2_bias': {'ln_idx': 1},
        'block_mlp_weight': {'mlp_idx': 0, 'conv1d_idx': 2},
        'block_mlp_bias': {'mlp_idx': 0, 'conv1d_idx': 2},
        'block_mlp_projection_weight': {'mlp_idx': 0, 'conv1d_idx': 3},
        'block_mlp_projection_bias': {'mlp_idx': 0, 'conv1d_idx': 3},
        'layer_norm_weight': {'ln_idx': 0},
        'layer_norm_bias': {'ln_idx': 0},
    }

    @staticmethod
    def _tf_idx(idx: int) -> str:
        return f'_{idx}' if idx > 0 else ''

    def _get_layer_idx(self, i: int, key: str) -> dict[str, str]:
        return {
            k: self._tf_idx(v * i + self.layer_idx_offset[key].get(k, 0))
            for k, v in self.layer_idx_multiplier.items()
        }

    def __init__(self, model_config: str, model: str | None = None):
        super().__init__()
        self.model = mcpt_tf.Model.from_config_(model_config, load_model=model)
        self.weights = {weight.name: weight for weight in self.model.weights}
        self.n_layer = self.model.config['n_layer']
        self.print_loaded_keys()

    def get_weight(self, key: str, **info) -> np.ndarray:
        info.update(self._get_layer_idx(info.get('i', 0), key))
        info.update({'i': self._tf_idx(info.get('i', 0))})
        return self.weights[self.weight_map[key].format(**info)].numpy()

    def set_weight(self, key: str, value: np.ndarray, **info):
        info.update(self._get_layer_idx(info.get('i', 0), key))
        info.update({'i': self._tf_idx(info.get('i', 0))})
        key_ = self.weight_map[key].format(**info)
        if list(value.shape) != list(self.weights[key_].shape):
            value = np.expand_dims(value, axis=0)
        if list(value.shape) != list(self.weights[key_].shape):
            raise_size_mismatch(key_, list(value.shape), list(self.weights[key_].shape))
        print(f'{key:<40} {str(list(value.shape)):<15} -> {key_:<40} {str(list(self.weights[key_].shape)):<15}')
        self.weights[key_].assign(tf.Variable(value, dtype=tf.float32))

    def save_model(self, save_path: str | None) -> str:
        self.model.set_weights(tf.keras.backend.batch_get_value(self.weights.values()))
        save_path = pathlib.Path(save_path).with_suffix('.h5')
        self.model.save_weights(save_path)
        return str(save_path)


class TorchModelManagerBase(ModelManager, abc.ABC):
    weight_map = {}

    def get_weight(self, key: str, **info) -> np.ndarray:
        return self.weights[self.weight_map[key].format(**info)].numpy()

    def set_weight(self, key: str, value: np.ndarray, **info):
        key_ = self.weight_map[key].format(**info)
        if list(value.shape) != list(self.weights[key_].shape):
            value = value[0]
        if list(value.shape) != list(self.weights[key_].shape):
            raise_size_mismatch(key_, list(value.shape), list(self.weights[key_].shape))
        print(f'{key:<40} {str(list(value.shape)):<15} -> {key_:<40} {str(list(self.weights[key_].shape)):<15}')
        self.weights[key_] = torch.from_numpy(value)


class TorchModelManager(TorchModelManagerBase):
    weight_map = {
        'word_token_embedding': 'transformer.wte.weight',
        'position_token_embedding': 'transformer.wpe.weight',
        'block_layer_norm_1_weight': 'transformer.blocks.{i}.ln_1.weight',
        'block_layer_norm_1_bias': 'transformer.blocks.{i}.ln_1.bias',
        'block_attention_weight': 'transformer.blocks.{i}.attn.c_attn.w',
        'block_attention_bias': 'transformer.blocks.{i}.attn.c_attn.b',
        'block_attention_projection_weight': 'transformer.blocks.{i}.attn.c_proj.w',
        'block_attention_projection_bias': 'transformer.blocks.{i}.attn.c_proj.b',
        'block_layer_norm_2_weight': 'transformer.blocks.{i}.ln_2.weight',
        'block_layer_norm_2_bias': 'transformer.blocks.{i}.ln_2.bias',
        'block_mlp_weight': 'transformer.blocks.{i}.mlp.c_fc.w',
        'block_mlp_bias': 'transformer.blocks.{i}.mlp.c_fc.b',
        'block_mlp_projection_weight': 'transformer.blocks.{i}.mlp.c_proj.w',
        'block_mlp_projection_bias': 'transformer.blocks.{i}.mlp.c_proj.b',
        'layer_norm_weight': 'transformer.ln_f.weight',
        'layer_norm_bias': 'transformer.ln_f.bias',
    }
    name = 'PyTorch'

    def __init__(self, model_config: str, model: str | None = None):
        super().__init__()
        self.model = linglong.Model.from_config(model_config, load_model=model, device='cpu')
        self.weights = self.model.state_dict()
        self.n_layer = self.model.config['n_layer']
        self.print_loaded_keys()

    def save_model(self, save_path: str | None) -> str:
        self.model.load_state_dict(self.weights)
        save_path = pathlib.Path(save_path).with_suffix('.pt')
        torch.save(self.model.state_dict(), save_path)
        return str(save_path)


class TransformersModelManager(TorchModelManagerBase):
    weight_map = {
        'word_token_embedding': 'wte.weight',
        'position_token_embedding': 'wpe.weight',
        'block_layer_norm_1_weight': 'h.{i}.ln_1.weight',
        'block_layer_norm_1_bias': 'h.{i}.ln_1.bias',
        'block_attention_weight': 'h.{i}.attn.c_attn.weight',
        'block_attention_bias': 'h.{i}.attn.c_attn.bias',
        'block_attention_projection_weight': 'h.{i}.attn.c_proj.weight',
        'block_attention_projection_bias': 'h.{i}.attn.c_proj.bias',
        'block_layer_norm_2_weight': 'h.{i}.ln_2.weight',
        'block_layer_norm_2_bias': 'h.{i}.ln_2.bias',
        'block_mlp_weight': 'h.{i}.mlp.c_fc.weight',
        'block_mlp_bias': 'h.{i}.mlp.c_fc.bias',
        'block_mlp_projection_weight': 'h.{i}.mlp.c_proj.weight',
        'block_mlp_projection_bias': 'h.{i}.mlp.c_proj.bias',
        'layer_norm_weight': 'ln_f.weight',
        'layer_norm_bias': 'ln_f.bias',
    }
    name = 'Transformers'

    def __init__(self, model_config: str | None = None, model: str | None = None):
        super().__init__()
        if model is not None:
            self.model = linglong.LingLongModel.from_pretrained(model)
        elif model_config is not None:
            self.model = linglong.LingLongModel(linglong.LingLongConfig.from_json_file(model_config))
        else:
            raise ValueError('Either model or model config must be provided.')
        self.weights = self.model.state_dict()
        self.n_layer = self.model.config.n_layer
        self.print_loaded_keys()

    def save_model(self, save_path: str | None) -> str:
        self.model.load_state_dict(self.weights)
        save_path = pathlib.Path(save_path).with_suffix('')
        self.model.save_pretrained(save_path)
        return str(save_path)


model_manager_map = {
    'tensorflow': TensorFlowModelManager,
    'torch': TorchModelManager,
    'transformers': TransformersModelManager,
}


def transfer_weights(src: ModelManager, dst: ModelManager, key: str, **info):
    weight = src.get_weight(key, **info)
    dst.set_weight(key, weight, **info)


def main(
        src_type: str,
        dst_type: str,
        src_model: str,
        dst_model_config: str,
        src_model_config: str | None = None,
        dst_model: str | None = None,
        vocab_path: str | None = None,
):
    src_model_manager = model_manager_map[src_type](src_model_config, src_model)
    dst_model_manager = model_manager_map[dst_type](dst_model_config)

    # Transfer word token embedding
    transfer_weights(src_model_manager, dst_model_manager, 'word_token_embedding')

    # Transfer position token embedding
    transfer_weights(src_model_manager, dst_model_manager, 'position_token_embedding')

    # Transfer block weights
    for i in range(src_model_manager.n_layer):
        transfer_weights(src_model_manager, dst_model_manager, 'block_layer_norm_1_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_layer_norm_1_bias', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_attention_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_attention_bias', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_attention_projection_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_attention_projection_bias', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_layer_norm_2_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_layer_norm_2_bias', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_mlp_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_mlp_bias', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_mlp_projection_weight', i=i)
        transfer_weights(src_model_manager, dst_model_manager, 'block_mlp_projection_bias', i=i)

    # Transfer layer norm weights
    transfer_weights(src_model_manager, dst_model_manager, 'layer_norm_weight', i=src_model_manager.n_layer)
    transfer_weights(src_model_manager, dst_model_manager, 'layer_norm_bias', i=src_model_manager.n_layer)

    dst_model_path = dst_model_manager.save_model(dst_model if dst_model else src_model)

    if dst_type == 'transformers':
        tokenizer = linglong.Tokenizer(vocab_path)
        tokenizer.save_pretrained(dst_model_path)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
