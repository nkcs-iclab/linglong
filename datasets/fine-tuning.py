import fire
import pathlib

from torch.utils.data import DataLoader

import linglong
import linglong.records


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        model_config: str,
        split: str = 'train',
        dataset_config: str = 'configs/fine-tuning/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: str = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        items_per_file: int = 200000,
        special_tokens: dict[str, str] | None = None,
        n_example: int = 3,
):
    with linglong.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '[MASK]',
            'end_token': '[CLS]',
            'part_separator': '[unused1]',
            'segment_separator': '[unused2]',
            **(special_tokens or {}),
        }
        model_config_path = model_config
        model_config = linglong.LingLongConfig.from_json_file(model_config_path)
        config = linglong.merge_configs({
            'dataset': dataset,
            'dataset_config_path': dataset_config,
            'model_config_path': model_config_path,
            'model_config': model_config,
            'input_path': input_path,
            'output_path': output_path,
            'split': split,
            'vocab': vocab,
            'pinyin_vocab': pinyin_vocab,
            'use_cache': use_cache,
            'items_per_file': items_per_file,
            'special_tokens': special_tokens,
        }, linglong.load_config(dataset_config, key=dataset))
        dataset_path = pathlib.Path(output_path) / dataset
        spinner.write(linglong.prettify(
            config,
            default=lambda o: {'n_positions': o.n_positions, 'use_pinyin': o.use_pinyin},
        ))

    with linglong.running(f'Processing {dataset} dataset', spinner=use_cache) as spinner:
        dataset = linglong.datasets.finetuning.load(config)
        meta_path, records_path = dataset.prepare()
        meta = linglong.load_config(meta_path)
        spinner.write(linglong.prettify({
            'meta_path': meta_path,
            'records_path': records_path,
            'meta': meta,
        }))

    print(linglong.text('Examples:', style=linglong.INFO))
    dataset = linglong.records.load(
        records_path,
        meta_path,
        use_pinyin=model_config.use_pinyin,
    )
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.Tokenizer(vocab)
    for batch in data_loader:
        linglong.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
