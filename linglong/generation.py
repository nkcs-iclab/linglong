class BasePlugin:

    def __call__(self, prompt: str) -> str | dict:
        raise NotImplementedError
