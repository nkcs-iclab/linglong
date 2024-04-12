import abc


class BasePlugin(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def call(self, prompt: str) -> str | dict:
        pass

    def __call__(self, prompt: str) -> str | dict:
        return self.call(prompt)
