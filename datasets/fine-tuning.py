import fire

from torch.utils.data import DataLoader

import linglong
import linglong.data.tfrecord


def main(
        dataset: str,
        input_path: str,
        output_path: str,
        dataset_config: str,
        vocab_path: str | None = None,
        pinyin_vocab_path: str | None = None,
        model_config: str | None = None,
        model_path: str | None = None,
        split: str = 'train',
        use_cache: bool = False,
        items_per_file: int = 200000,
        special_tokens: dict[str, str] | None = None,
        n_example: int = 3,
):
    with linglong.running('Loading configs') as spinner:
        special_tokens = {
            'part_separator': '<unused1>',
            'segment_separator': '<unused2>',
            **(special_tokens or {}),
        }
        model_config_path = model_config
        if model_path is not None:
            model_config = linglong.LingLongConfig.from_pretrained(model_path)
        elif model_config_path is not None:
            model_config = linglong.LingLongConfig.from_json_file(model_config_path)
        else:
            raise ValueError('Either `model_config` or `model_path` should be provided.')
        config = linglong.merge_configs({
            'dataset': dataset,
            'input_path': input_path,
            'output_path': output_path,
            'split': split,
            'vocab_path': vocab_path,
            'pinyin_vocab_path': pinyin_vocab_path,
            'special_tokens': special_tokens,
            'use_cache': use_cache,
            'items_per_file': items_per_file,
            'use_pinyin': model_config.use_pinyin,
            'n_position': model_config.n_position,
            'model_path': model_path,
            'model_config_path': model_config_path,
            'dataset_config_path': dataset_config,
        }, linglong.load_config(dataset_config, key=dataset))
        spinner.write(linglong.prettify(config))

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
    dataset = linglong.data.tfrecord.load_tfrecord_dataset(
        records_path,
        meta_path,
        use_pinyin=model_config.use_pinyin,
    )
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.get_tokenizers(
        vocab_path=vocab_path,
        pretrained_model=model_path,
        special_tokens=special_tokens,
    )[0]
    for batch in data_loader:
        linglong.data.print_model_inputs(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
