import math
import torch

from typing import *


class CosineAnnealingWarmup(torch.optim.lr_scheduler.LambdaLR):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int,
            decay_steps: int,
            alpha: float = 0.1,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps - warmup_steps
        self.alpha = alpha
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1.0, self.decay_steps))
        return max(self.alpha, 0.5 * (1.0 + math.cos(math.pi * progress)))


def cosine_annealing_warmup(
        optimizer,
        config: Dict[str, Any],
        n_ctx: int,
        dp_size: int = 1,
        alpha: float = 0.1,
        last_epoch: int = -1,
):
    warmup_steps = int(config['warmup_tokens'] / config['batch_size'] / dp_size / n_ctx)
    decay_steps = int(config['decay_tokens'] / config['batch_size'] / dp_size / n_ctx)
    return CosineAnnealingWarmup(
        optimizer,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        alpha=alpha,
        last_epoch=last_epoch,
    )
