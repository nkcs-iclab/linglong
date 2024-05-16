import pathlib
import requests

import linglong


class Plugin(linglong.generation.BasePlugin):

    def __init__(self):
        super().__init__()
        self.config = linglong.merge_configs(
            linglong.load_config(str(pathlib.Path(__file__).parent / 'configs' / 'ollama.yaml')),
            linglong.load_config(str(pathlib.Path(__file__).parent / 'configs' / 'ollama.local.yaml')),
        )
        self.placeholder = self.config.get('placeholder', 'OLLAMA')

    def call(self, prompt: str) -> str | dict:
        if prompt.strip() == '':
            return ''
        params = {
            'model': self.config['model'],
            'messages': [
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            'stream': False,
        }
        response = requests.post(
            self.config['endpoint'],
            json=params,
        )
        response.raise_for_status()
        print(response.json())
        return response.json()['message']['content']
