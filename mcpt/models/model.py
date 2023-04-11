import torch
import torch.nn as nn

from typing import *


class Model(nn.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = self.transformer.config

    def forward(self, inputs, past=None):
        return self.transformer(inputs, past=past)

    @staticmethod
    def from_config(
            config: Union[str, Dict[str, Any]],
            load_model: Optional[str] = None,
            device: Optional = None,
    ) -> 'Model':
        import mcpt
        if isinstance(config, str):
            config = mcpt.load_config(config)
        if config.get('use_pinyin', False):
            model = Model(mcpt.models.MCPTPinyinModel(config))
        else:
            model = Model(mcpt.models.MCPTModel(config))
        if load_model is not None:
            model.load_state_dict(torch.load(load_model, map_location=device))
        if device is not None:
            model.to(device)
        return model
