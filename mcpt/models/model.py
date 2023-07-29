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

    @classmethod
    def from_config(
            cls,
            config: Union[str, Dict[str, Any]],
            load_model: Optional[str] = None,
            device: Optional = None,
            strict: bool = True,
    ) -> 'Self':
        import mcpt
        if isinstance(config, str):
            config = mcpt.load_config(config)
        if config.get('use_pinyin', False):
            model = cls(mcpt.models.MCPTPinyinModel(config))
        else:
            model = cls(mcpt.models.MCPTModel(config))
        if load_model is not None:
            model.load_state_dict(torch.load(load_model, map_location=device), strict=strict)
        if device is not None:
            model.to(device)
        return model


class RewardModel(Model):

    def __init__(self, transformer):
        super().__init__(transformer)
        self.reward_head = nn.Linear(self.config['n_vocab'], 1, bias=False)

    def forward(self, inputs, past=None):
        logits = super().forward(inputs, past=past)['logits']
        rewards = self.reward_head(logits).squeeze(-1)
        return rewards
