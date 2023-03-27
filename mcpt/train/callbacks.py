import torch
import pathlib

from typing import *


class ModelCheckpointCallback:

    def __init__(
            self,
            save_path: pathlib.Path,
            save_frequency: Union[int, str],
            has_validation_data: bool = False,
    ):
        self.save_path = save_path
        self.save_frequency = save_frequency
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_name = 'E{epoch}'
        if self.save_frequency != 'epoch':
            self.save_name += 'B{batch}'
        self.save_name += 'L{loss:.6f}'
        if self.save_frequency == 'epoch' and has_validation_data:
            self.save_name += 'VL{val_loss:.6f}'
        self.save_name += '.pt'
        self.last_epoch = 1

    def __call__(self, model, epoch: int, loss: float, batch: Optional[int] = None, val_loss: Optional[float] = None):
        save_name = self.save_name.format(epoch=epoch, batch=batch, loss=loss, val_loss=val_loss)
        if self.save_frequency == 'epoch':
            if epoch == self.last_epoch:
                return
            self.last_epoch = epoch
        if isinstance(self.save_frequency, int):
            if batch is None or batch % self.save_frequency != 0:
                return
        torch.save(model.state_dict(), str(self.save_path.joinpath(save_name)))
