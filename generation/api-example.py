import fire
import torch

from peft import PeftModelForCausalLM

import linglong


def main(
        model: str,
        prompt: str,
        peft_model: str | None = None,
        vocab: str | None = '../common/vocab/char-13312.txt',
        pinyin_vocab: str | None = '../common/vocab/pinyin-1354.txt',
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
        special_tokens: dict[str, str] | None = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
):
    model_path = model
    special_tokens = {
        'start_token': '[MASK]',
        'end_token': '[CLS]',
        'part_separator': '[unused1]',
        'segment_separator': '[unused2]',
        'new_line': '[SEP]',
        **(special_tokens or {}),
    }
    model = linglong.LingLongLMHeadModel.from_pretrained(model_path, device_map=device_map)
    if peft_model is not None:
        model = PeftModelForCausalLM.from_pretrained(model, peft_model, device_map=device_map)
    tokenizer, pinyin_tokenizer = linglong.load_tokenizer(
        vocab_path=vocab,
        pinyin_vocab_path=pinyin_vocab,
        pretrained_model=model_path,
        special_tokens=special_tokens,
        use_pinyin=model.config.use_pinyin,
        padding_side='left',
    )

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
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    generated_text = tokenizer.batch_decode(generated_ids)[0]
    print(generated_text)


if __name__ == '__main__':
    linglong.init()
    fire.Fire(main)
