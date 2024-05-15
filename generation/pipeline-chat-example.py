import fire
import torch

from transformers import AutoTokenizer, pipelines


def main(
        model: str,
        prompt: str,
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
        special_tokens: dict[str, str] | None = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if special_tokens is not None and len(new_special_tokens := list(
            set(special_tokens.values()) - set(tokenizer.all_special_tokens),
    )) > 0:
        tokenizer.add_special_tokens({
            'additional_special_tokens': new_special_tokens,
        })
    pipeline = pipelines.pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device_map,
        trust_remote_code=True,
    )
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    generated_text = pipeline(messages, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
    print(generated_text[0]['generated_text'][-1]['content'])


if __name__ == '__main__':
    fire.Fire(main)
