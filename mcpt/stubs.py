from typing import *


class Horovod:

    def __init__(self, size: int = 1, rank: int = 0, local_rank: int = 0):
        self._size = size
        self._rank = rank
        self._local_rank = local_rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def local_rank(self) -> int:
        return self._local_rank

    def init(self):
        pass

    def broadcast_parameters(self, *args, **kwargs):
        pass

    def broadcast_optimizer_state(self, *args, **kwargs):
        pass

    # noinspection PyPep8Naming
    @staticmethod
    def DistributedOptimizer(optimizer, **_):
        return optimizer


class Writable:

    def __init__(self, print_fn: Optional[Callable] = print):
        self._print_fn = print_fn

    def write(self, text_):
        self._print_fn(text_)
