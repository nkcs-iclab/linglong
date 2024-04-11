import fire
import pathlib

from torch.utils.data import DataLoader

import linglong


def main(
        dataset: str,
        input_path: str,
        model_config: str,
        vocab_path: str,
        pinyin_vocab_path: str | None = None,
        stride: int | None = None,
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
        model_config_path = model_config
        model_config = linglong.LingLongConfig.from_json_file(model_config_path)
        input_path = pathlib.Path(input_path) / dataset
        config = linglong.datasets.pretraining.StreamingPreTrainingDatasetConfig(
            input_path=input_path,
            vocab_path=vocab_path,
            special_tokens=special_tokens,
            stride=stride or model_config.n_position // 2,
            n_position=model_config.n_position,
            use_pinyin=model_config.use_pinyin,
            pinyin_vocab_path=pinyin_vocab_path,
        )
        spinner.write(linglong.prettify(config))

    print(linglong.text('Examples:', style=linglong.INFO))
    dataset = linglong.datasets.pretraining.StreamingPreTrainingDataset(config)
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.get_tokenizers(
        vocab_path=vocab_path,
        special_tokens=special_tokens,
    )[0]
    for batch in data_loader:
        linglong.data.print_model_inputs(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
