import torch
from typing import Optional
from torch import Tensor


class MaskedSparseCategoricalCrossentropy(torch.nn.module):
    def __init__(self, reduction: str = 'none') -> None:
        super(MaskedSparseCategoricalCrossentropy, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, sample_weight=None) -> Tensor:
        losses = torch.nn.functional.cross_entropy(input=input, target=target, reduction=self.reduction)
        losses *= sample_weight
        loss = torch.sum(losses) / torch.sum(sample_weight)
        return loss
        return

# __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
# ignore_index: int
# label_smoothing: float
#
# def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
#              reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
#     super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
#     self.ignore_index = ignore_index
#     self.label_smoothing = label_smoothing
#
# def forward(self, input: Tensor, target: Tensor) -> Tensor:
#     return F.cross_entropy(input, target, weight=self.weight,
#                            ignore_index=self.ignore_index, reduction=self.reduction,
#                            label_smoothing=self.label_smoothing)
