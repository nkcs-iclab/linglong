import fire
import torch
import pathlib
import contextlib

from typing import *

import mcpt
import mcpt.records

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    hvd = mcpt.stubs.Horovod()


def main(
        training_data: str,
        model_config: str,
        training_config: str,
        override_config: Optional[Dict] = None,
        training_meta: str = 'train-meta.json',
        validation_data: Optional[str] = None,
        validation_meta: str = 'valid-meta.json',
        epochs: int = 20,
        load_model: Optional[str] = None,
        save_path: str = './ckpt',
        save_frequency: Union[int, str, List[Union[int, str]]] = 'epoch',
        log_frequency: int = 10,
        device: str = 'cuda',
):
    hvd.init()
    mcpt.bind_gpu(hvd)

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
            'validation_data': validation_data,
            'validation_meta': validation_meta,
            'epochs': epochs,
            'load_model': load_model,
            'save_path': save_path,
            'save_frequency': save_frequency,
            'log_frequency': log_frequency,
            'device': device,
            'model_config': model_config,
            'training_config': training_config,
        }
        save_path = pathlib.Path(save_path)
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running('Loading the dataset', hvd=hvd, timer=True):
        train_chosen_loader, train_rejected_loader = mcpt.records.load_rlhf(
            path=training_data,
            meta=training_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            stage=2,
        )
        validation_chosen_loader, validation_rejected_loader = mcpt.records.load_rlhf(
            path=validation_data,
            meta=validation_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            stage=2,
        )

    with mcpt.running('Loading the model', hvd=hvd, timer=True):
        model = mcpt.RewardModel.from_config(
            config=model_config,
            load_model=load_model,
            device=device,
            strict=False,
        )
        training_config['optimizer']['params']['lr'] = \
            training_config['optimizer']['params']['lr'] * hvd.size() * training_config['gradient_accumulation_steps']
        optimizer = mcpt.train.optimizers.adamw(model.parameters(), config=training_config['optimizer']['params'])
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            sparse_as_dense=True,
            backward_passes_per_step=training_config['gradient_accumulation_steps'],
        )
        lr_scheduler = mcpt.train.schedulers.cosine_annealing_warmup(
            optimizer,
            config=training_config,
            n_ctx=model.config['n_ctx'],
            dp_size=hvd.size(),
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        callbacks = []
        if hvd.rank() == 0:
            if not isinstance(save_frequency, Union[Tuple, List]):
                save_frequency = (save_frequency,)
            for freq in save_frequency:
                if isinstance(freq, str):
                    assert freq in ('epoch', 'step')
                callbacks.append(
                    mcpt.train.callbacks.ModelCheckpointCallback(
                        save_path=save_path,
                        save_frequency=freq,
                    ),
                )

    if hvd.rank() == 0:
        print(model)

    scaler = torch.cuda.amp.GradScaler() if training_config['fp16']['enabled'] else mcpt.stubs.Noop()
    for epoch in range(1, epochs + 1):
        if hvd.rank() == 0:
            print(f'Epoch {epoch}/{epochs}')
        train_tqdm = mcpt.tqdm(
            enumerate(zip(train_chosen_loader, train_rejected_loader)),
            total=len(train_chosen_loader),
            hvd=hvd,
        )
        loss = 0
        for batch_idx, (chosen, rejected) in train_tqdm:
            chosen = chosen[0].to(device)
            rejected = rejected[0].to(device)
            data = torch.cat((chosen, rejected), dim=0)
            optimizer.zero_grad()
            with (torch.cuda.amp.autocast() if training_config['fp16']['enabled'] else contextlib.suppress()):
                rewards = model(data)
                loss = mcpt.train.losses.reward_loss(inputs=data, rewards=rewards)['loss']
            if training_config['fp16']['enabled']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if not isinstance(hvd, mcpt.stubs.Horovod):
                optimizer.synchronize()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config['gradient_clipping'])
            with (contextlib.suppress() if isinstance(hvd, mcpt.stubs.Horovod) else optimizer.skip_synchronize()):
                if training_config['fp16']['enabled']:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            scaler.update()
            lr_scheduler.step()
            loss = loss.item()
            if hvd.rank() == 0 and (batch_idx + 1) % log_frequency == 0:
                train_tqdm.write(
                    f'Train Epoch: {epoch}/{epochs} [{batch_idx + 1}/{len(train_chosen_loader)}] '
                    f'Loss: {loss}' +
                    (f' Loss Scale: {scaler.get_scale()}' if training_config['fp16']['enabled'] else ''),
                )
            for callback in callbacks:
                callback(model=model, epoch=epoch, batch=batch_idx + 1, loss=loss)
        if validation_data is not None:
            model.eval()
            correct_predictions = 0
            total_predictions = 0
            scores = 0
            batch_idx = 0
            validation_tqdm = mcpt.tqdm(
                enumerate(zip(validation_chosen_loader, validation_rejected_loader)),
                total=99,
                hvd=hvd,
            )
            for batch_idx, (chosen, rejected) in validation_tqdm:
                chosen = chosen[0].to(device)
                rejected = rejected[0].to(device)
                data = torch.cat((chosen, rejected), dim=0)
                with torch.no_grad():
                    rewards = model(data)
                val_loss = mcpt.train.losses.reward_loss(inputs=data, rewards=rewards)
                chosen_mean_scores = val_loss['chosen_mean_scores']
                rejected_mean_scores = val_loss['rejected_mean_scores']
                correct_predictions += (chosen_mean_scores > rejected_mean_scores).sum()
                total_predictions += chosen_mean_scores.shape[0]
                scores += val_loss['chosen_mean_scores'].mean().float()
                if batch_idx == 99:
                    break
            acc = correct_predictions / total_predictions
            scores = scores / (batch_idx + 1)
            try:
                acc = hvd.allreduce(acc, name='acc').item()
                scores = hvd.allreduce(scores, name='scores').item()
            except:
                print(mcpt.text('Failed to allreduce.', mcpt.WARNING))
                pass
            if hvd.rank() == 0:
                print(f'Valid Epoch: [1/1] chosen_last_scores: {scores} acc: {acc}')
            model.train()

        for callback in callbacks:
            callback(
                model=model,
                epoch=epoch,
                loss=loss,
                val_loss=acc if validation_data is not None else None,
                end_of_epoch=True,
            )


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
