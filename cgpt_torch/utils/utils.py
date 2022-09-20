import importlib.util
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def is_torch_available():
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False

    return _torch_available
