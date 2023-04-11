import torch

from typing import *


def adamw(
        params,
        config: Dict[str, Any],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params,
        lr=config['lr'],
        betas=config['betas'],
        eps=config['eps'],
        weight_decay=config['weight_decay'],
    )
