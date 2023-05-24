import fire
import torch
import pathlib
import deepspeed

from typing import *

import mcpt
import mcpt.records


def int_or_str(value: str) -> Union[int, str]:
    try:
        return int(value)
    except ValueError:
        return value


def cosine_annealing_warmup(
        config: Dict[str, Any],
        n_ctx: int,
        dp_size: int = 1,
        alpha: float = 0.1,
        last_epoch: int = -1,
) -> Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]:
    def _get_lr_scheduler(optimizer_):
        return mcpt.train.schedulers.cosine_annealing_warmup(
            optimizer_,
            config=config,
            n_ctx=n_ctx,
            dp_size=dp_size,
            alpha=alpha,
            last_epoch=last_epoch,
        )

    return _get_lr_scheduler


class DSModelCheckpointCallback(mcpt.train.callbacks.ModelCheckpointCallback):

    def __init__(self, zero_optimization: bool = False, hvd: Optional = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zero_optimization = zero_optimization
        self._rank = hvd.rank() if hvd is not None else 0

    def __call__(
            self,
            model: deepspeed.DeepSpeedEngine,
            epoch: int,
            loss: float,
            batch: Optional[int] = None,
            val_loss: Optional[float] = None,
            end_of_epoch: bool = False,
    ):
        save_name = self._get_save_name(batch=batch, val_loss=val_loss)
        save_name = save_name.format(epoch=epoch, batch=batch, loss=loss, val_loss=val_loss)
        if self.save_frequency == 'epoch' and not end_of_epoch:
            return
        if isinstance(self.save_frequency, int):
            if end_of_epoch or batch % self.save_frequency != 0:
                return
        model.save_checkpoint(str(self.save_path), tag=save_name.split('L')[0])


def main(
        training_data: str,
        model_config: str,
        training_config: str,
        override_config: Optional[Dict] = None,
        training_meta: str = 'train-meta.pkl',
        epochs: int = 5,
        load_model: Optional[str] = None,
        save_path: str = './ckpt',
        save_frequency: Union[int, str, List[Union[int, str]]] = 'epoch',
        log_frequency: int = 10,
        local_rank: int = 0,
):
    deepspeed.init_distributed()
    hvd = mcpt.stubs.Horovod(
        size=deepspeed.comm.get_world_size(),
        rank=deepspeed.comm.get_rank(),
        local_rank=local_rank,
    )

    with mcpt.running('Loading configs', hvd=hvd) as spinner:
        model_config_path, training_config_path = model_config, training_config
        model_config = mcpt.load_config(model_config_path)
        model_config = mcpt.merge_configs(model_config, (override_config or {}).get('model_config', {}))
        training_config = mcpt.load_config(training_config_path)
        training_config = mcpt.merge_configs(training_config, (override_config or {}).get('training_config', {}))
        config = {
            'training_data': training_data,
            'model_config_path': model_config_path,
            'training_config_path': training_config_path,
            'override_config': override_config,
            'training_meta': training_meta,
            'epochs': epochs,
            'load_model': load_model,
            'save_path': save_path,
            'save_frequency': save_frequency,
            'model_config': model_config,
            'training_config': training_config,
            'log_frequency': log_frequency,
        }
        save_path = pathlib.Path(save_path)
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running('Loading the dataset', hvd=hvd, timer=True):
        train_loader = mcpt.records.load(
            path=training_data,
            meta=training_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=model_config.get('use_pinyin', False),
        )

    with mcpt.running('Loading the model', hvd=hvd, timer=True):
        model = mcpt.Model.from_config(
            config=model_config,
            load_model=load_model,
        )

        lr_scheduler = cosine_annealing_warmup(
            config=training_config,
            n_ctx=model.config['n_ctx'],
            dp_size=hvd.size(),
        )
        # noinspection PyTypeChecker
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            lr_scheduler=lr_scheduler,
            config=training_config,
        )
        callbacks = []
        if not isinstance(save_frequency, Union[Tuple, List]):
            save_frequency = (save_frequency,)
        for freq in save_frequency:
            if isinstance(freq, str):
                assert freq in ('epoch', 'step')
            callbacks.append(
                DSModelCheckpointCallback(
                    zero_optimization=training_config['zero_optimization']['stage'] > 0,
                    hvd=hvd,
                    save_path=save_path,
                    save_frequency=freq,
                ),
            )

    if hvd.rank() == 0:
        print(model)

    for epoch in range(1, epochs + 1):
        train_tqdm = mcpt.tqdm(enumerate(train_loader), total=len(train_loader), hvd=hvd)
        loss = 0
        for batch_idx, (data, target) in train_tqdm:
            data, target = data.to('cuda'), target.to('cuda')
            logits, present = model_engine(data)
            loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
            model_engine.backward(loss)
            model_engine.step()
            if hvd.rank() == 0 and (batch_idx + 1) % log_frequency == 0:
                train_tqdm.write(
                    f'Train Epoch: {1}/{epochs} [{batch_idx + 1}/{len(train_loader)}] '
                    f'Loss: {loss.item()}'
                )
            for callback in callbacks:
                callback(model=model_engine, epoch=epoch, batch=batch_idx + 1, loss=loss)
        for callback in callbacks:
            callback(model=model_engine, epoch=epoch, loss=loss, end_of_epoch=True)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
