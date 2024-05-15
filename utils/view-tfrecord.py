import fire

from torch.utils.data import DataLoader

import linglong
import linglong.data.tfrecord


def main(
        path: str,
        meta: str,
        vocab_path: str,
        use_pinyin: bool = False,
        n_example: int = 3,
):
    dataset = linglong.data.tfrecord.load_tfrecord_dataset(path, meta, use_pinyin=use_pinyin)
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.get_tokenizers(vocab_path=vocab_path)
    linglong.data.print_model_inputs(next(iter(data_loader)), tokenizer=tokenizer)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
