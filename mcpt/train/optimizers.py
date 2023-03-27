import torch

from typing import *


def adamw_warmup_cosine_decay(
        params,
        config: Dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=config['lr'],
        betas=(config['beta_1'], config['beta_2']),
        eps=config['epsilon'],
        weight_decay=config['weight_decay_rate'],
    )
