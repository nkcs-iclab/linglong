import fire
import torch

import mcpt


def main(
        model: str,
        prompt: str,
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
):
    model_path = model
    model = mcpt.LingLongLMHeadModel.from_pretrained(model_path, device_map=device_map)
    tokenizer = mcpt.Tokenizer.from_pretrained(model_path, padding_side='left')
    pinyin_tokenizer = mcpt.PinyinTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        fallback=tokenizer,
    ) if model.config.use_pinyin else None

    model_inputs = tokenizer([prompt], return_tensors='pt', padding=True).to(model.device)
    if pinyin_tokenizer:
        model_inputs['pinyin_input_ids'] = pinyin_tokenizer(
            [prompt],
            return_tensors='pt',
            padding=True,
        ).to(model.device)['input_ids']
        if model_inputs['input_ids'].shape[1] != model_inputs['pinyin_input_ids'].shape[1]:
            raise ValueError('The length of the pinyin input must be the same as the text input.')

    generated_ids = model.generate(
        **model_inputs,
        max_length=128,
        do_sample=True,
        top_k=20,
        top_p=1,
        temperature=1,
    )
    generated_text = tokenizer.batch_decode(generated_ids)[0]
    print(generated_text)


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
