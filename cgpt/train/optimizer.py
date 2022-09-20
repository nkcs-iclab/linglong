from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import *
import torch
import math


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_decay_steps: int,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
        return max(0.1, 0.5 * (1.0 + torch.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimizer(model_parameters, params: Dict[str, Any], n_ctx: int,
                  data_parallel_size: int = 1) -> Optimizer:
    warmup_steps = int(params['warmup_tokens'] / params['batch_size'] / data_parallel_size / n_ctx)
    decay_steps = int(params['decay_tokens'] / params['batch_size'] / data_parallel_size / n_ctx)
    base_optimizer = AdamW(model_parameters, lr=params['lr'], betas=(params['beta_1'], params['beta_2']),
                           eps=params['epsilon'], weight_decay=params['weight_decay_rate'],
                           correct_bias=params['correct_bias'])
    optimizer = get_cosine_schedule_with_warmup(optimizer=base_optimizer, num_warmup_steps=warmup_steps,
                                                num_decay_steps=decay_steps)
    return optimizer
