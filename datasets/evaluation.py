import fire

from torch.utils.data import DataLoader

import linglong


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        dataset_config: str,
        vocab_path: str | None = None,
        pinyin_vocab_path: str | None = None,
        use_cache: bool = False,
        special_tokens: dict[str, str] | None = None,
        n_example: int = 3,
):
    with linglong.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '<|startoftext|>',
            'end_token': '<|endoftext|>',
            'part_separator': '<unused1>',
            'segment_separator': '<unused2>',
            **(special_tokens or {}),
        }

        config = linglong.merge_configs({
            'dataset': dataset,
            'input_path': input_path,
            'output_path': output_path,
            'dataset_config_path': dataset_config,
            'vocab_path': vocab_path,
            'pinyin_vocab_path': pinyin_vocab_path,
            'use_cache': use_cache,
            'special_tokens': special_tokens,
        }, linglong.load_config(dataset_config, key=dataset))
        model_path = config['model'] if isinstance(config['model'], str) else config['model']['base']
        model_config = linglong.LingLongConfig.from_pretrained(model_path)
        config['use_pinyin'] = model_config.use_pinyin
        tokenizer = linglong.get_tokenizers(
            vocab_path=vocab_path,
            pretrained_model=model_path,
            special_tokens=special_tokens,
        )[0]
        spinner.write(linglong.prettify(config))

    with linglong.running(f'Loading {dataset} dataset', spinner=use_cache):
        dataset = linglong.datasets.evaluation.load(config)
        data, candidates = dataset.prepare()
        meta = {
            'count': len(data),
            'candidates': candidates,
        }
        spinner.write(linglong.prettify(meta))

    print(linglong.text('Examples:', style=linglong.INFO))
    dataset = linglong.data.DictDataset(data)
    data_loader = DataLoader(dataset, batch_size=n_example, collate_fn=linglong.data.padded_batch)
    for batch in data_loader:
        linglong.data.print_model_inputs(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
