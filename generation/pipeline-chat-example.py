import fire
import torch

from transformers import pipeline


def main(
        model: str,
        prompt: str,
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
):
    pipe = pipeline(
        'text-generation',
        model=model,
        device=device_map,
        trust_remote_code=True,
    )
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    output = pipe(messages, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
    print(output)


if __name__ == '__main__':
    fire.Fire(main)
