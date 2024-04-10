import os
import cmd
import fire
import torch
import importlib

from typing import Callable
from transformers import TextStreamer
from transformers.utils import logging
from peft import PeftModelForCausalLM

import linglong


class LingLongGenerate(cmd.Cmd):

    def __init__(
            self,
            generation_config: dict,
            tokenizer: linglong.Tokenizer,
            pinyin_tokenizer: linglong.PinyinTokenizer | None,
            model: linglong.Model,
            special_tokens: dict[str, str],
            prompt: str,
            prefix: str,
            suffix: str,
            plugins: list[Callable] | None = None,
            debug: bool = False,
    ):
        super().__init__()
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.pinyin_tokenizer = pinyin_tokenizer
        self.model = model
        self.use_pinyin = self.model.config.use_pinyin
        self.special_tokens = special_tokens
        self.llm_prompt = prompt
        self.prefix = prefix
        self.suffix = suffix
        self.plugins = plugins or []
        self.debug = debug
        self._renew_cmd_prompt()
        self.use_rawinput = True
        # noinspection PyTypeChecker
        self.streamer = TextStreamer(self.tokenizer)

    def _renew_cmd_prompt(self):
        prompt = f'{self.llm_prompt[:10]}...' if len(self.llm_prompt) > 10 else self.llm_prompt
        self.prompt = linglong.text(f'({prompt})', style=linglong.ERROR)
        self.prompt += f' [max length: {linglong.text(self.generation_config["max_length"], style=linglong.STRUCTURE)}'
        if self.prefix:
            self.prompt += f', prefix: {linglong.text(self.prefix, style=linglong.STRUCTURE)}'
        if self.suffix:
            self.prompt += f', suffix: {linglong.text(self.suffix, style=linglong.STRUCTURE)}'
        self.prompt += '] -> '

    def _print_samples(self, samples: list[str]):
        for idx, sample in enumerate(samples):
            sample = sample.split(self.special_tokens['new_line'] + self.special_tokens['end_token'])[0]
            sample = sample.split(self.special_tokens['new_line'])[0]
            sample = sample.replace(self.special_tokens['new_line'], '\\n')
            print(linglong.text(f'GENERATED [{idx + 1}]', style=linglong.WARNING), end=' ')
            print(sample)

    def _generate(self):
        backward = self.model.config.backward
        step_by_step = self.generation_config['batch_size'] == 1 and not backward

        print(linglong.text('QUERY', style=linglong.WARNING), self.llm_prompt)

        prefix = self.prefix
        for plugin in self.plugins:
            if '{' + plugin.placeholder + '}' in self.prefix:
                plugin_output = plugin(self.llm_prompt)
                if isinstance(plugin_output, dict):
                    plugin_output, debug_output = plugin_output['text'], plugin_output['debug']
                    for k, v in debug_output.items():
                        print(linglong.text(f'PLUGIN {plugin.placeholder} - {k}', style=linglong.WARNING), v)
                print(linglong.text(f'PLUGIN {plugin.placeholder}', style=linglong.WARNING), plugin_output)
                prefix = prefix.replace('{' + plugin.placeholder + '}', plugin_output)
        prompt = self.special_tokens['start_token'] + prefix + (
            self.llm_prompt[::-1] if backward else self.llm_prompt) + self.suffix
        if self.debug:
            print(linglong.text('PROMPT', style=linglong.WARNING), prompt)
        model_inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.model.device)
        if self.use_pinyin:
            model_inputs['pinyin_input_ids'] = self.pinyin_tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
            ).to(self.model.device)['input_ids']
        try:
            if step_by_step:
                print(linglong.text(f'GENERATED', style=linglong.WARNING), end=' ')
                _ = self.model.generate(
                    **model_inputs,
                    max_length=self.generation_config['max_length'],
                    do_sample=True,
                    top_k=self.generation_config['top_k'],
                    top_p=self.generation_config['top_p'],
                    temperature=self.generation_config['temperature'],
                    streamer=self.streamer,
                )
            else:
                with linglong.running(f'Generating {self.generation_config["batch_size"]} sample(s)', timer=True):
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_length=self.generation_config['max_length'],
                        do_sample=True,
                        top_k=self.generation_config['top_k'],
                        top_p=self.generation_config['top_p'],
                        temperature=self.generation_config['temperature'],
                        num_return_sequences=self.generation_config['batch_size'],
                    )
                    if backward:
                        generated_ids = torch.flip(generated_ids, [1])
                    generated_text = self.tokenizer.batch_decode(generated_ids)
        except KeyboardInterrupt:
            print()

        if not step_by_step:
            self._print_samples(generated_text)

    def do_set(self, arg):
        allowed_keys = {*self.generation_config.keys(), 'prefix', 'suffix'}
        k, v = arg.split()
        if k not in allowed_keys:
            print(linglong.text(f'`{k}` is not a valid key. Valid keys are: {allowed_keys}', style=linglong.ERROR))
            return
        if k in ('max_length', 'batch_size', 'top_k'):
            v = int(v)
        elif k in ('temperature', 'top_p'):
            v = float(v)
        elif k == 'prefix':
            self.prefix = v
        elif k == 'suffix':
            self.suffix = v
        if k == 'max_length' and v > self.model.config.n_positions:
            v = self.model.config.n_positions
            logger.warning(
                f'The max generation length cannot be set to {v}. '
                f'Clipping the length to {self.model.config.n_positions}.',
            )
        if k in self.generation_config:
            self.generation_config[k] = v
        print(linglong.text(f'`{k}` is set to {v}', style=linglong.INFO))
        self._renew_cmd_prompt()

    def do_clear(self, arg):
        if len(arg) == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            if arg == 'prefix':
                self.prefix = ''
            elif arg == 'suffix':
                self.suffix = ''
            self._renew_cmd_prompt()

    @staticmethod
    def do_exit(_):
        print(linglong.text('Goodbye', style=linglong.INFO))
        return True

    def emptyline(self):
        self._generate()

    def default(self, line: str):
        self.llm_prompt = line.strip()
        self._renew_cmd_prompt()
        self._generate()


def main(
        model: str,
        peft_model: str | None = None,
        vocab_path: str | None = None,
        pinyin_vocab_path: str | None = None,
        batch_size: int = 1,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        device_map: str | dict[str, int | str | torch.device] | int | torch.device | None = 'cuda',
        special_tokens: dict[str, str] | None = None,
        prompt: str = '齐小明，科学家',
        prefix: str = '',
        suffix: str = '',
        plugins: list[str] | None = None,
        plugin_kwargs: dict[str, dict[str, str]] | None = None,
        debug: bool = False,
):
    generation_config = {
        'batch_size': batch_size,
        'max_length': max_length,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
    }
    special_tokens = {
        'start_token': '<|startoftext|>',
        'end_token': '<|endoftext|>',
        'part_separator': '<unused1>',
        'segment_separator': '<unused2>',
        'new_line': '<sep>',
        **(special_tokens or {}),
    }
    try:
        model_path = model
        with linglong.running('Loading the model', timer=True):
            model = linglong.LingLongLMHeadModel.from_pretrained(model_path, device_map=device_map)
            if peft_model is not None:
                model = PeftModelForCausalLM.from_pretrained(model, peft_model, device_map=device_map)
        tokenizer, pinyin_tokenizer = linglong.get_tokenizers(
            vocab_path=vocab_path,
            pinyin_vocab_path=pinyin_vocab_path,
            pretrained_model=model_path,
            special_tokens=special_tokens,
            use_pinyin=model.config.use_pinyin,
            padding_side='left',
        )
        if max_length > model.config.n_positions:
            max_length = model.config.n_positions
            logger.warning(
                f'The max generation length cannot be set to {max_length}. '
                f'Clipping the length to {model.config.n_positions}.',
            )
        if plugins is not None:
            plugins = [importlib.import_module(plugin).Plugin() for plugin in plugins]
        print(linglong.text('Model Info', style=linglong.INFO))
        print(linglong.prettify({
            'model': model_path,
            **({'peft_model': peft_model} if peft_model is not None else {}),
            'use_pinyin': model.config.use_pinyin,
            'backward': model.config.backward,
        }))

        LingLongGenerate(
            generation_config=generation_config,
            tokenizer=tokenizer,
            pinyin_tokenizer=pinyin_tokenizer,
            model=model,
            special_tokens=special_tokens,
            prompt=prompt,
            prefix=prefix,
            suffix=suffix,
            plugins=plugins,
            debug=debug,
        ).cmdloop()
    except KeyboardInterrupt:
        print(linglong.text('\nGoodbye', style=linglong.INFO))


if __name__ == '__main__':
    linglong.init()
    logger = logging.get_logger()
    fire.Fire(main)
