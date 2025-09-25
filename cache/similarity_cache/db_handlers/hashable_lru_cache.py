from functools import lru_cache, wraps
from typing import Any, Callable, Tuple

import numpy as np


def _freeze(obj: Any) -> Any:
    """Convert common unhashable objects to hashable, recursively."""
    # Fast path for already-hashable scalars
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return obj

    # Built-ins
    if isinstance(obj, list):
        return "__list__", tuple(_freeze(x) for x in obj)
    if isinstance(obj, tuple):
        return "__tuple__", tuple(_freeze(x) for x in obj)
    if isinstance(obj, set):
        # sort for determinism
        return "__set__", tuple(sorted(_freeze(x) for x in obj))
    if isinstance(obj, dict):
        # sort by key for determinism
        return "__dict__", tuple(sorted((_freeze(k), _freeze(v)) for k, v in obj.items()))

    # Pydantic models (v2)
    if hasattr(obj, "model_dump"):
        return "__pyd__", _freeze(obj.model_dump())

    # NumPy arrays (optional)
    if np is not None and isinstance(obj, np.ndarray):
        return "__nd__", obj.dtype.str, tuple(obj.shape), obj.tobytes()

    # Generic iterables (best-effort)
    if hasattr(obj, "__iter__"):
        return "__iter__", tuple(_freeze(x) for x in obj)

    # Fallback: hope it's hashable
    return obj


def _thaw(obj: Any) -> Any:
    """Reconstruct a usable Python object for the user function."""
    if not (isinstance(obj, tuple) and obj and isinstance(obj[0], str) and obj[0].startswith("__")):
        return obj

    tag = obj[0]
    if tag == "__list__":
        return [_thaw(x) for x in obj[1]]
    if tag == "__tuple__":
        return tuple(_thaw(x) for x in obj[1])
    if tag == "__set__":
        return set(_thaw(x) for x in obj[1])
    if tag == "__dict__":
        return {_thaw(k): _thaw(v) for (k, v) in obj[1]}
    if tag == "__pyd__":
        return _thaw(obj[1])
    if tag == "__nd__" and np is not None:
        _, dtype_str, shape, buf = obj
        arr = np.frombuffer(buf, dtype=np.dtype(dtype_str))
        return arr.reshape(shape)
    if tag == "__iter__":
        return tuple(_thaw(x) for x in obj[1])

    return obj


def hashable_lru_cache(func: Callable[..., Any] | None = None, *, maxsize: int | None = 128, typed: bool = False):
    """
    Decorator factory: like functools.lru_cache, but accepts unhashable args/kwargs
    (lists, dicts, sets, numpy arrays, pydantic models, …) by freezing them into
    hashable keys transparently.

    Usage:
        @hashable_lru_cache(maxsize=256)
        def embed_many(texts: list[str], model: str = "text-embedding-3-small"):
            ...

    Notes:
      - Arguments are *logically* the same; the cache key is built from their frozen forms.
      - The wrapped function is called with thawed objects (lists restored as lists, etc.).

    """

    def _make_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @lru_cache(maxsize=maxsize, typed=typed)
        def _cached(frozen_args: Tuple[Any, ...], frozen_kwargs: Tuple[Tuple[Any, Any], ...]):
            args = tuple(_thaw(x) for x in frozen_args)
            kwargs = {_thaw(k): _thaw(v) for (k, v) in frozen_kwargs}
            return fn(*args, **kwargs)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            f_args = tuple(_freeze(a) for a in args)
            f_kwargs = tuple(sorted((_freeze(k), _freeze(v)) for k, v in kwargs.items()))
            return _cached(f_args, f_kwargs)

        # expose cache controls like functools.lru_cache does
        wrapper.cache_info = _cached.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = _cached.cache_clear  # type: ignore[attr-defined]
        return wrapper

    # bare decorator usage: @hashable_lru_cache
    if callable(func):
        return _make_decorator(func)

    # called with params: @hashable_lru_cache(...)
    return _make_decorator
