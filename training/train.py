import fire
import torch
import pathlib

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
        extra_config: Optional[Dict] = None,
        training_meta: str = 'train-meta.pkl',
        validation_data: Optional[str] = None,
        validation_meta: str = 'valid-meta.pkl',
        use_pinyin: bool = False,
        load_model: Optional[str] = None,
        save_path: str = './ckpt',
        save_frequency: Union[int, str, Tuple[Union[int, str], ...]] = 'epoch',
        device: str = 'cuda',
):
    hvd.init()
    mcpt.bind_gpu(hvd)

    with mcpt.running('Loading configs', hvd=hvd) as spinner:
        config = {
            'training-data': training_data,
            'model-config': model_config,
            'training-config': training_config,
            'extra-config': extra_config,
            'training-meta': training_meta,
            'validation-data': validation_data,
            'validation-meta': validation_meta,
            'use-pinyin': use_pinyin,
            'load-model': load_model,
            'save-path': save_path,
            'save-frequency': save_frequency,
        }
        model_config = mcpt.load_config(model_config)
        model_config = mcpt.merge_configs(model_config, (extra_config or {}).get('model', {}))
        training_config = mcpt.load_config(training_config)
        training_config = mcpt.merge_configs(training_config, (extra_config or {}).get('training', {}))
        save_path = pathlib.Path(save_path)
        config['model-config-dict'] = model_config
        config['training-config-dict'] = training_config
        spinner.write(mcpt.print_dict(config, export=True))

    with mcpt.running('Loading the dataset', hvd=hvd, timer=True):
        train_loader = mcpt.records.load(
            path=training_data,
            meta=training_meta,
            batch_size=training_config['batch_size'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=use_pinyin,
        )
        validation_loader = mcpt.records.load(
            path=validation_data,
            meta=validation_meta,
            batch_size=training_config['batch_size'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=use_pinyin,
        )

    with mcpt.running('Loading the model', hvd=hvd, timer=True):
        model_config, model = mcpt.create_model_from_config(
            model_config,
            load_model=load_model,
            use_pinyin=use_pinyin,
        )
        model.to(device)
        training_config['lr'] = training_config['lr'] * hvd.size() * training_config['backward_passes_per_step']
        optimizer = mcpt.train.optimizers.adamw(model.parameters(), config=training_config)
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            sparse_as_dense=True,
            backward_passes_per_step=training_config['backward_passes_per_step'],
        )
        lr_scheduler = mcpt.train.schedulers.cosine_annealing_warmup(
            optimizer,
            config=training_config,
            n_ctx=model_config['n_ctx'],
            dp_size=hvd.size(),
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        callbacks = []
        if hvd.rank() == 0:
            if not isinstance(save_frequency, tuple):
                save_frequency = (save_frequency,)
            for freq in save_frequency:
                if isinstance(freq, str):
                    assert freq in ('epoch', 'step')
                callbacks.append(mcpt.train.callbacks.ModelCheckpointCallback(
                    save_path=save_path,
                    save_frequency=freq,
                    has_validation_data=validation_data is not None),
                )

    if hvd.rank() == 0:
        print(model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, training_config['epochs'] + 1):
        if hvd.rank() == 0:
            print(f'Epoch {epoch}/{training_config["epochs"]}')
        train_tqdm = mcpt.tqdm(enumerate(train_loader), total=len(train_loader), hvd=hvd)
        loss = 0
        for batch_idx, (data, target) in train_tqdm:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, present = model(data)
                loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
            scaler.scale(loss).backward()
            optimizer.synchronize()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config['clip_norm'])
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            loss = loss.item()
            if hvd.rank() == 0:
                train_tqdm.write(
                    f'Train Epoch: {epoch}/{training_config["epochs"]} [{batch_idx + 1}/{len(train_loader)}] '
                    f'Loss: {loss} Loss Scale: {scaler.get_scale()}'
                )
            for callback in callbacks:
                callback(model=model, epoch=epoch, batch=batch_idx + 1, loss=loss)
        if validation_data is not None:
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):
                    data, target = data.to(device), target.to(device)
                    logits, present = model(data)
                    val_loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
                    if hvd.rank() == 0:
                        print(f'Valid Epoch: [{batch_idx + 1}/{len(validation_loader)}] Loss: {val_loss.item()}')
            model.train()
        for callback in callbacks:
            callback(
                model=model,
                epoch=epoch,
                batch=None,
                loss=loss,
                val_loss=val_loss.item() if validation_data is not None else None,
            )


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
