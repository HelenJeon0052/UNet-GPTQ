def __getattr__(name):
    if name not in ("reconstruct_pair", "reconstruct_adjacent_aggregate"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import samplers
    value = getattr(samplers, name)
    globals()[name] = value
    return value


__all__ = ["reconstruct_pair", "reconstruct_adjacent_aggregate"]
__version__ = '0.1.0'
