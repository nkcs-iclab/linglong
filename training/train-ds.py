import torch
import pathlib
import argparse
import deepspeed

from typing import *
from deepspeed.utils import zero_to_fp32

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
        if self._rank == 0:
            if self._zero_optimization:
                zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(
                    checkpoint_dir=str(self.save_path),
                    output_file=str(self.save_path / save_name),
                    tag=save_name.split('L')[0],
                )
            else:
                torch.save(model.module.state_dict(), str(self.save_path / save_name))


def main(cmd_args: argparse.Namespace):
    hvd = mcpt.stubs.Horovod(
        size=deepspeed.comm.comm.get_world_size(),
        rank=deepspeed.comm.comm.get_rank(),
        local_rank=deepspeed.comm.comm.get_local_rank(),
    )
    with mcpt.running('Loading configs', hvd=hvd) as spinner:
        model_config = mcpt.load_config(cmd_args.model_config)
        training_config = mcpt.load_config(cmd_args.training_config)
        save_path = pathlib.Path(cmd_args.save_path)
        spinner.write(mcpt.print_dict(cmd_args.__dict__, export=True))
        spinner.write(mcpt.print_dict(model_config, export=True))

    with mcpt.running('Loading the dataset', hvd=hvd, timer=True):
        train_loader = mcpt.records.load(
            path=cmd_args.training_data,
            meta=cmd_args.training_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=model_config.get('use_pinyin', False),
        )

    with mcpt.running('Loading the model', hvd=hvd, timer=True):
        model = mcpt.Model.from_config(
            config=model_config,
            load_model=cmd_args.load_model,
        )

        lr_scheduler = cosine_annealing_warmup(
            config=training_config,
            n_ctx=model.config['n_ctx'],
            dp_size=hvd.size(),
        )
        # noinspection PyTypeChecker
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            model_parameters=model.parameters(),
            lr_scheduler=lr_scheduler,
            config=training_config,
        )
        callbacks = []
        for freq in cmd_args.save_frequency:
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

    for epoch in range(1, cmd_args.epochs + 1):
        train_tqdm = mcpt.tqdm(enumerate(train_loader), total=len(train_loader), hvd=hvd)
        loss = 0
        for batch_idx, (data, target) in train_tqdm:
            data, target = data.to('cuda'), target.to('cuda')
            logits, present = model_engine(data)
            loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
            model_engine.backward(loss)
            model_engine.step()
            if hvd.rank() == 0 and (batch_idx + 1) % cmd_args.log_frequency == 0:
                train_tqdm.write(
                    f'Train Epoch: {1}/{cmd_args.epochs} [{batch_idx + 1}/{len(train_loader)}] '
                    f'Loss: {loss.item()}'
                )
            for callback in callbacks:
                callback(model=model_engine, epoch=epoch, batch=batch_idx + 1, loss=loss)
        for callback in callbacks:
            callback(model=model_engine, epoch=epoch, loss=loss, end_of_epoch=True)


if __name__ == '__main__':
    mcpt.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--training_config', type=str, required=True)
    parser.add_argument('--training_data', type=str, required=True)
    parser.add_argument('--training_meta', type=str, default='train-meta.pkl')
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--save_path', type=str, default='./ckpt')
    parser.add_argument('--log_frequency', type=int, default=10)
    parser.add_argument('--save_frequency', type=int_or_str, nargs='+', default=['epoch'])
    parser.add_argument('--epochs', type=int, default=5)
    parser = deepspeed.add_config_arguments(parser)
    deepspeed.init_distributed()
    main(parser.parse_args())
