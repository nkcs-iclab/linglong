import fire

import mcpt


def main(
        model: str,
        model_config: str,
        prompt: str,
        vocab: str = '../common/vocab/char-13312.txt',
        device: str = 'cuda',
):
    model_config, model = mcpt.create_model_from_config(
        model_config=model_config,
        load_model=model,
        device=device,
    )
    tokenizer = mcpt.tokenization.Tokenizer(vocab)
    generated = mcpt.generation.generate(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
    )
    print(generated)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
