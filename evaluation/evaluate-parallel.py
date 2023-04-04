import fire
import math
import pickle
import concurrent.futures as futures

from typing import *

import mcpt
import mcpt.evaluation


def _split_process(n: int, workers: List[int]) -> List[int]:
    workers = [w / sum(workers) for w in workers]
    split = [math.floor(n * w) for w in workers]
    left = n - sum(split)
    i = 0
    while left > 0:
        split[i % len(split)] += 1
        left -= 1
        i += 1
    return split


def work(
        x,
        y_true,
        candidates: Optional[List[str]],
        config: Dict[str, Any],
        pid: int,
        offset: int,
        tokenizer: mcpt.tokenization.Tokenizer,
        pinyin_tokenizer: mcpt.tokenization.PinyinTokenizer,
        special_token_ids: Dict[str, int],
        device: str,
        callbacks: Optional[List[Callable]] = None,
):
    eval_fn = mcpt.evaluation.get_eval_fn(config.get('evaluation-method', 'generation'))
    model = mcpt.models.Model.from_config(
        config=config['model-config'],
        load_model=config['model'],
        use_pinyin=config['use-pinyin'],
        device=device,
    )
    y_pred = eval_fn(
        x=x,
        model=model,
        offset=offset,
        y_true=y_true,
        config=config,
        tokenizer=tokenizer,
        candidates=candidates,
        pinyin_tokenizer=pinyin_tokenizer,
        special_token_ids=special_token_ids,
        device=device,
        callbacks=callbacks,
    )
    return pid, y_pred


def main(
        dataset: str,
        model_config: str,
        input_path: str,
        cache_path: str,
        workers: str,
        dataset_config: str = 'configs/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str = '../common/vocab/pinyin-1354.txt',
        use_pinyin: bool = False,
        use_cache: bool = False,
        output_path_template: str = '{dataset}-{model}-{split}-{template-id}-{timestamp}',
        special_tokens: Optional[Dict[str, str]] = None,
        verbose: int = 2,
        slicer: Optional[str] = None,
        items_per_process: Optional[int] = None,
        device: str = 'cuda',
        **kwargs,
):
    with mcpt.running('Loading configs') as spinner:
        special_tokens = {
            'start-token': '[MASK]',
            'end-token': '[CLS]',
            'part-separator': '[unused1]',
            'segment-separator': '[unused2]',
            **(special_tokens or {}),
        }
        config = mcpt.merge_configs(
            {
                'dataset': dataset,
                'dataset-config': dataset_config,
                'model-config': model_config,
                'input-path': input_path,
                'cache-path': cache_path,
                'vocab': vocab,
                'pinyin-vocab': pinyin_vocab,
                'use-pinyin': use_pinyin,
                'use-cache': use_cache,
                'output-path-template': output_path_template,
                'special-tokens': special_tokens,
                'verbose': verbose,
                'slicer': slicer,
                'workers': workers,
                'items-per-process': items_per_process,
                'device': device,
            },
            mcpt.load_config(dataset_config, key=dataset),
            kwargs,
        )

        tokenizer = mcpt.tokenization.Tokenizer(vocab)
        pinyin_tokenizer = mcpt.tokenization.PinyinTokenizer(vocab_file=pinyin_vocab, fallback=tokenizer)
        output_path = mcpt.evaluation.get_output_path(config)
        config['output-path'] = output_path
        special_token_ids = {
            key: tokenizer.convert_tokens_to_ids(value) for (key, value) in special_tokens.items()
        }
        eval_metric = mcpt.evaluation.get_eval_metric(config.get('evaluation-metric'))
        spinner.write(mcpt.print_dict(config, export=True))

    with mcpt.running(f'Loading {dataset} dataset', spinner=use_cache):
        x, y_true, candidates = mcpt.evaluation.load_dataset(config)
        if slicer is not None:
            slicer = slice(*(int(x) if x else None for x in slicer.split(':')))
            x, y_true = x[slicer], y_true[slicer]
        candidates = tokenizer.convert_tokens_to_ids(candidates) if candidates is not None else None

    with mcpt.running(f'Setting up workers') as spinner:
        worker_config = {}
        workers = str(workers).split(',')
        n_workers = 0
        for worker in workers:
            if ':' in worker:
                gpu_id, gpu_workers = worker.split(':')
                worker_config[int(gpu_id)] = {'workers': int(gpu_workers), 'processes': 0}
                n_workers += int(gpu_workers)
            else:
                worker_config[0] = {'workers': int(worker), 'processes': 0}
                n_workers += int(worker)
        worker_config = [[k, v] for k, v in worker_config.items()]
        if items_per_process is None:
            n_process = n_workers
            items_per_process = math.ceil(len(x) / n_process)
        else:
            n_process = math.ceil(len(x) / items_per_process)
        process_split = _split_process(n_process, [worker[1]['workers'] for worker in worker_config])
        for processes, worker in zip(process_split, worker_config):
            worker[1]['processes'] = processes
        worker_config = {k: v for k, v in worker_config}
        spinner.write(mcpt.print_dict(worker_config, export=True))

    print(mcpt.text('Evaluating', style=mcpt.INFO))
    pid = 0
    y_pred = []
    futures_ = []
    executors = []
    manager = mcpt.evaluation.CallbackManager()
    manager.start()
    lock = manager.Lock()
    # noinspection PyUnresolvedReferences
    progress_bar_callback = manager.ProgressBarCallback(len(x), lock=lock)
    # noinspection PyUnresolvedReferences
    log_result_callback = manager.LogResultCallback(
        path=output_path,
        verbose=verbose,
        tokenizer=tokenizer,
        use_perplexity=config.get('use-perplexity', False),
        use_pinyin=use_pinyin,
        lock=lock,
    )
    for gpu_id, worker in worker_config.items():
        device = f'cuda:{gpu_id}' if gpu_id > -1 else 'cpu'
        # noinspection PyArgumentList
        executor = futures.ProcessPoolExecutor(
            max_workers=worker['workers'],
            max_tasks_per_child=1,
        )
        executors.append(executor)
        for _ in range(worker['processes']):
            if len(x[pid * items_per_process: (pid + 1) * items_per_process]) > 0:
                futures_.append(executor.submit(
                    work,
                    pid=pid,
                    config=config,
                    tokenizer=tokenizer,
                    candidates=candidates,
                    offset=pid * items_per_process,
                    pinyin_tokenizer=pinyin_tokenizer,
                    special_token_ids=special_token_ids,
                    x=x[pid * items_per_process: (pid + 1) * items_per_process],
                    y_true=y_true[pid * items_per_process: (pid + 1) * items_per_process],
                    device=device,
                    callbacks=[
                        progress_bar_callback,
                        log_result_callback,
                    ],
                ))
                pid += 1
    for future in futures.as_completed(futures_):
        y_pred.append(future.result())
    progress_bar_callback.close()
    manager.shutdown()
    for executor in executors:
        executor.shutdown()
    y_pred.sort(key=lambda _: _[0])
    y_pred = [b for _, a in y_pred for b in a]

    if verbose > 0:
        with mcpt.running('Saving intermediate results'):
            with open(output_path + '.pkl', 'wb') as f:
                pickle.dump(config, f)
                pickle.dump(y_pred, f)

    if eval_metric is not None:
        print(mcpt.text('Calculating evaluation metrics', style=mcpt.INFO))
        result = eval_metric(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            config=config,
            tokenizer=tokenizer,
            output_path=output_path,
            special_token_ids=special_token_ids,
        )
        print(f'{config["evaluation-metric"]}: {mcpt.print_dict(result)}')
    else:
        print(mcpt.text('No evaluation metric is specified.', style=mcpt.WARNING))


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
