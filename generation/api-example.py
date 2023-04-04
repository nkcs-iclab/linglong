import fire

import mcpt


def main(
        model: str,
        model_config: str,
        prompt: str,
        vocab: str = '../common/vocab/char-13312.txt',
        device: str = 'cuda',
):
    model = mcpt.Model.from_config(
        config=model_config,
        load_model=model,
        device=device,
    )
    model.eval()
    tokenizer = mcpt.Tokenizer(vocab)
    generated = mcpt.generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )
    print(generated)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
