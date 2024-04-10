import pathlib
import requests

from peft import PeftModelForCausalLM

import linglong


class Plugin(linglong.generation.BasePlugin):

    def __init__(self):
        super().__init__()
        self.config = linglong.merge_configs(
            linglong.load_config(str(pathlib.Path(__file__).parent / 'configs' / 'bingsearch.yaml')),
            linglong.load_config(str(pathlib.Path(__file__).parent / 'configs' / 'bingsearch.local.yaml')),
        )
        self.placeholder = self.config.get('placeholder', 'BINGSEARCH')
        if self.config.get('extract_keywords', False):
            self.keywords_model = linglong.LingLongLMHeadModel.from_pretrained(
                self.config['keywords_model']['base'],
                device_map=self.config['keywords_model']['device_map'],
            )
            if self.config['keywords_model'].get('peft') is not None:
                self.keywords_model = PeftModelForCausalLM.from_pretrained(
                    self.keywords_model,
                    self.config['keywords_model']['peft'],
                    device_map=self.config['keywords_model']['device_map'],
                )
            self.special_tokens = self.config['keywords_model'].get('special_tokens')
            self.template = self.config['keywords_model'].get('template', '{prompt}')
            self.keywords_tokenizer = linglong.get_tokenizers(
                vocab_path=self.config['keywords_model'].get('vocab_path'),
                pretrained_model=self.config['keywords_model']['base'],
                special_tokens=self.special_tokens,
            )[0]

    def __call__(self, prompt: str) -> str | dict:
        if prompt.strip() == '':
            return ''
        keywords_input = None
        if self.config.get('extract_keywords', False):
            keywords_input = self.template.format(prompt=prompt, **(self.special_tokens or {}))
            input_ids = self.keywords_tokenizer(
                keywords_input,
                return_tensors='pt',
            ).to(self.keywords_model.device)['input_ids']
            generated_ids = self.keywords_model.generate(
                input_ids,
                max_length=self.config['keywords_model']['max_length'],
                do_sample=False,
            )[0][input_ids.shape[1]:]
            prompt = self.keywords_tokenizer.decode(generated_ids, skip_special_tokens=True)
        params = {'q': prompt, 'mkt': self.config['mkt']}
        headers = {'Ocp-Apim-Subscription-Key': self.config['subscription_key']}
        response = requests.get(
            self.config['endpoint'],
            headers=headers,
            params=params,
            proxies=self.config.get('proxies'),
        )
        response.raise_for_status()
        result = response.json()['webPages']['value'][0]['snippet'] if 'webPages' in response.json() else ''
        if self.config.get('extract_keywords', False):
            return {
                'text': result,
                'debug': {
                    'KEYWORDS INPUT': keywords_input,
                    'KEYWORDS OUTPUT': prompt,
                },
            }
        return result
