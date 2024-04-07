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
        load_attention_mask: bool = True,
):
    dataset = linglong.records.load(path, meta, use_pinyin=use_pinyin, load_attention_mask=load_attention_mask)
    data_loader = DataLoader(dataset, batch_size=n_example)
    tokenizer = linglong.Tokenizer(vocab)
    for batch in data_loader:
        linglong.print_training_records(batch, tokenizer=tokenizer)
        break


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
