import os
import cmd
import fire
import torch
import warnings
import importlib

from typing import Callable
from transformers import TextStreamer

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
        self._generation_config = generation_config
        self._tokenizer = tokenizer
        self._pinyin_tokenizer = pinyin_tokenizer
        self._model = model
        self._use_pinyin = self._model.config.use_pinyin
        self._special_tokens = special_tokens
        self._prompt = prompt
        self._prefix = prefix
        self._suffix = suffix
        self._plugins = plugins or []
        self._debug = debug
        self._renew_cmd_prompt()
        self.use_rawinput = True
        # noinspection PyTypeChecker
        self._streamer = TextStreamer(self._tokenizer)

    def _renew_cmd_prompt(self):
        prompt = f'{self._prompt[:10]}...' if len(self._prompt) > 10 else self._prompt
        self.prompt = linglong.text(f'({prompt})', style=linglong.ERROR)
        self.prompt += f' [max length: {linglong.text(self._generation_config["max_length"], style=linglong.STRUCTURE)}'
        if self._prefix:
            self.prompt += f', prefix: {linglong.text(self._prefix, style=linglong.STRUCTURE)}'
        if self._suffix:
            self.prompt += f', suffix: {linglong.text(self._suffix, style=linglong.STRUCTURE)}'
        self.prompt += '] -> '

    def _print_samples(self, samples: list[str]):
        for idx, sample in enumerate(samples):
            sample = sample.split(self._special_tokens['new_line'] + self._special_tokens['end_token'])[0]
            sample = sample.split(self._special_tokens['new_line'])[0]
            sample = sample.replace(self._special_tokens['new_line'], '\\n')
            print(linglong.text(f'GENERATED [{idx + 1}]', style=linglong.WARNING), end=' ')
            print(sample)

    def _generate(self):
        backward = self._model.config.backward
        step_by_step = self._generation_config['batch_size'] == 1 and not backward

        print(linglong.text('QUERY', style=linglong.WARNING), self._prompt)

        prefix = self._prefix
        for plugin in self._plugins:
            if '{' + plugin.placeholder + '}' in self._prefix:
                plugin_output = plugin(self._prompt)
                if isinstance(plugin_output, dict):
                    plugin_output, debug_output = plugin_output['text'], plugin_output['debug']
                    print(debug_output)
                    for k, v in debug_output.items():
                        print(linglong.text(f'PLUGIN {plugin.placeholder} - {k}', style=linglong.WARNING), v)
                print(linglong.text(f'PLUGIN {plugin.placeholder}', style=linglong.WARNING), plugin_output)
                prefix = prefix.replace('{' + plugin.placeholder + '}', plugin_output)
        prompt = self._special_tokens['start_token'] + prefix + (
            self._prompt[::-1] if backward else self._prompt) + self._suffix
        if self._debug:
            print(linglong.text('PROMPT', style=linglong.WARNING), prompt)
        model_inputs = self._tokenizer([prompt], return_tensors='pt', padding=True).to(self._model.device)
        if self._use_pinyin:
            model_inputs['pinyin_input_ids'] = self._pinyin_tokenizer(
                [prompt],
                return_tensors='pt',
                padding=True,
            ).to(self._model.device)['input_ids']
        try:
            if step_by_step:
                print(linglong.text(f'GENERATED', style=linglong.WARNING), end=' ')
                _ = self._model.generate(
                    **model_inputs,
                    max_length=self._generation_config['max_length'],
                    do_sample=True,
                    top_k=self._generation_config['top_k'],
                    top_p=self._generation_config['top_p'],
                    temperature=self._generation_config['temperature'],
                    streamer=self._streamer,
                )
            else:
                with linglong.running(f'Generating {self._generation_config["batch_size"]} sample(s)', timer=True):
                    generated_ids = self._model.generate(
                        **model_inputs,
                        max_length=self._generation_config['max_length'],
                        do_sample=True,
                        top_k=self._generation_config['top_k'],
                        top_p=self._generation_config['top_p'],
                        temperature=self._generation_config['temperature'],
                        num_return_sequences=self._generation_config['batch_size'],
                    )
                    if backward:
                        generated_ids = torch.flip(generated_ids, [1])
                    generated_text = self._tokenizer.batch_decode(generated_ids)
        except KeyboardInterrupt:
            print()

        if not step_by_step:
            self._print_samples(generated_text)

    def do_set(self, arg):
        allowed_keys = {*self._generation_config.keys(), 'prefix', 'suffix'}
        k, v = arg.split()
        if k not in allowed_keys:
            print(linglong.text(f'`{k}` is not a valid key. Valid keys are: {allowed_keys}', style=linglong.ERROR))
            return
        if k in ('max_length', 'batch_size', 'top_k'):
            v = int(v)
        elif k in ('temperature', 'top_p'):
            v = float(v)
        elif k == 'prefix':
            self._prefix = v
        elif k == 'suffix':
            self._suffix = v
        if k == 'max_length' and v > self._model.config.n_positions:
            v = self._model.config.n_positions
            warnings.warn(f'The max generation length cannot be set to {v}. '
                          f'Clipping the length to {self._model.config.n_positions}.')
        if k in self._generation_config:
            self._generation_config[k] = v
        print(linglong.text(f'`{k}` is set to {v}', style=linglong.INFO))
        self._renew_cmd_prompt()

    def do_clear(self, arg):
        if len(arg) == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            if arg == 'prefix':
                self._prefix = ''
            elif arg == 'suffix':
                self._suffix = ''
            self._renew_cmd_prompt()

    @staticmethod
    def do_exit(_):
        print(linglong.text('Goodbye', style=linglong.INFO))
        return True

    def emptyline(self):
        self._generate()

    def default(self, line: str):
        self._prompt = line.strip()
        self._renew_cmd_prompt()
        self._generate()


def main(
        model: str,
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
        'start_token': '[MASK]',
        'end_token': '[CLS]',
        'part_separator': '[unused1]',
        'segment_separator': '[unused2]',
        'new_line': '[SEP]',
        **(special_tokens or {}),
    }
    try:
        model_path = model
        with linglong.running('Loading the model', timer=True):
            model = linglong.LingLongLMHeadModel.from_pretrained(model_path, device_map=device_map)
        tokenizer = linglong.Tokenizer.from_pretrained(model_path, padding_side='left')
        pinyin_tokenizer = linglong.PinyinTokenizer.from_pretrained(
            model_path,
            fallback=tokenizer,
            padding_side='left',
        ) if model.config.use_pinyin else None
        if max_length > model.config.n_positions:
            max_length = model.config.n_positions
            warnings.warn(f'The max generation length cannot be set to {max_length}. '
                          f'Clipping the length to {model.config.n_positions}.')
        if plugins is not None:
            plugins = [importlib.import_module(plugin).Plugin() for plugin in plugins]
        print(linglong.text('Model Info', style=linglong.INFO))
        print(linglong.prettify({
            'model': model_path,
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
    fire.Fire(main)
