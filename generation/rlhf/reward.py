import os
import cmd
import fire
import torch

from typing import *

import mcpt


class MCPTGenerate(cmd.Cmd):

    def __init__(
            self,
            tokenizer: mcpt.Tokenizer,
            model: mcpt.Model,
            device: str,
            special_tokens: Dict[str, str],
            prompt: str,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._special_tokens = special_tokens
        self._prompt = prompt

        self._renew_cmd_prompt()
        self.use_rawinput = True

    def _renew_cmd_prompt(self):
        prompt = f'{self._prompt[:10]}...' if len(self._prompt) > 10 else self._prompt
        self.prompt = mcpt.text(f'({prompt})', style=mcpt.ERROR)
        self.prompt += ' -> '

    def _print_scores(self, scores: torch.Tensor):
        print(mcpt.text('SCORES', style=mcpt.WARNING))
        print(scores)
        print(mcpt.text('RESPONSE SCORE', style=mcpt.WARNING))
        print(scores[0, -1].item())

    def _generate(self):
        print(mcpt.text('QUERY', style=mcpt.WARNING), self._prompt)

        prompt_ids = mcpt.generation.convert_prompt_to_ids(
            prompt=self._prompt,
            tokenizer=self._tokenizer,
            special_tokens=self._special_tokens,
        )
        try:
            with mcpt.running(f'Evaluating', timer=True):
                with torch.no_grad():
                    scores = self._model(torch.tensor([prompt_ids], device=self._device))
        except KeyboardInterrupt:
            print()

        self._print_scores(scores)

    def do_clear(self, _):
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def do_exit(_):
        print(mcpt.text('Goodbye', style=mcpt.INFO))
        return True

    def emptyline(self):
        self._generate()

    def default(self, line: str):
        self._prompt = line.strip()
        self._renew_cmd_prompt()
        self._generate()


def main(
        model: str,
        model_config: str,
        vocab: str = '../common/vocab/char-13312.txt',
        device: str = 'cuda',
        special_tokens: Optional[Dict[str, str]] = None,
        prompt: str = '齐小明，科学家',
):
    load_model = model
    special_tokens = {
        'start_token': '[MASK]',
        'end_token': '[CLS]',
        'part_separator': '[unused1]',
        'segment_separator': '[unused2]',
        'new_line': '[SEP]',
        **(special_tokens or {}),
    }
    try:
        with mcpt.running('Loading configs'):
            model_config = mcpt.load_config(model_config)
            tokenizer = mcpt.Tokenizer(vocab)

        with mcpt.running('Loading the model', timer=True):
            model = mcpt.RewardModel.from_config(
                config=model_config,
                load_model=load_model,
                device=device,
            )
            model.eval()

        MCPTGenerate(
            tokenizer=tokenizer,
            model=model,
            device=device,
            special_tokens=special_tokens,
            prompt=prompt,
        ).cmdloop()
    except KeyboardInterrupt:
        print(mcpt.text('\nGoodbye', style=mcpt.INFO))


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)
