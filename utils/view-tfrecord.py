import fire

from torch.utils.data import DataLoader

import linglong
import linglong.records


def main(
        path: str,
        meta: str,
        use_pinyin: bool = False,
        vocab: str = '../common/vocab/char-13312.txt',
        n_example: int = 3,
):
    dataset = linglong.records.load(path, meta, use_pinyin=use_pinyin)
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.Tokenizer(vocab)
    linglong.print_training_records(next(iter(data_loader)), tokenizer=tokenizer)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
