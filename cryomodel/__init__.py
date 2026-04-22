"""CryoModel: unified cryo-EM modeling toolkit."""

try:
    import importlib.metadata as _im

    __version__ = _im.version("cryomodel")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
