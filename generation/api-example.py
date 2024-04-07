import fire
import torch

import linglong


def main(
        model: str,
        prompt: str,
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
):
    model_path = model
    model = linglong.LingLongLMHeadModel.from_pretrained(model_path, device_map=device_map)
    tokenizer = linglong.Tokenizer.from_pretrained(model_path, padding_side='left')
    pinyin_tokenizer = linglong.PinyinTokenizer.from_pretrained(
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
    linglong.init()
    fire.Fire(main)
