"""Remembering model answers, so a re-run costs nothing.

SUPPLIED — you do not need to change this file.

Labs 3 and its variants call the clip analyser repeatedly over the same
flagged moments. Each call is a video request, so without this the second run
of an agent costs exactly as much as the first. Caching to disk rather than
memory means it survives restarting Streamlit, which happens a lot in a lab.

This is a quota mitigation wearing the clothes of a convenience. Twenty-five
people re-running agents against six sessions is the load that gets close to a
shared rate limit; almost all of it is repeat work.

    @disk_cached("notes")
    def analyze_clip(...): ...

Delete `.cache/` to force fresh answers — worth doing when you change a prompt,
since the cache key covers the arguments and not your instructions.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from lab.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".cache"


def _key(name: str, args: tuple, kwargs: dict) -> Path:
    payload = json.dumps([name, args, sorted(kwargs.items())], default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
    return CACHE_DIR / name / f"{digest}.json"


def _return_model(func: Callable) -> Any:
    """The Pydantic model a function returns, if it returns one.

    Reads it through ``get_type_hints`` rather than ``__annotations__``. Under
    ``from __future__ import annotations`` — which every module here uses —
    annotations are plain strings, so the raw dict yields ``"TechniqueNotes"``
    and the cache would hand back dicts instead of models.
    """
    try:
        import typing

        hint = typing.get_type_hints(func).get("return")
    except Exception:
        return None
    return hint if hasattr(hint, "model_validate") else None


def disk_cached(name: str) -> Callable:
    """Cache a function's return value on disk, keyed by its arguments.

    The wrapped function must return something JSON-serialisable, or a Pydantic
    model — those are stored as their ``model_dump()`` and rebuilt on the way
    out.

    Args:
        name: subdirectory under ``.cache/``, so different functions do not
            collide.
    """

    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = _key(name, args, kwargs)
            model = _return_model(func)
            if path.is_file():
                try:
                    stored = json.loads(path.read_text())
                    if hasattr(model, "model_validate"):
                        return model.model_validate(stored)
                    return stored
                except Exception:
                    path.unlink(missing_ok=True)  # corrupt entry, just redo it

            result = func(*args, **kwargs)
            try:
                payload = (
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else result
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, default=str, indent=1))
            except Exception:
                pass  # a cache that cannot write must not break the call
            return result

        wrapper.cache_dir = CACHE_DIR / name  # type: ignore[attr-defined]
        return wrapper

    return decorate


def clear(name: str | None = None) -> int:
    """Remove cached answers.

    Args:
        name: only this subdirectory, or every one when omitted.

    Returns:
        How many entries were removed.
    """
    target = CACHE_DIR / name if name else CACHE_DIR
    if not target.exists():
        return 0
    removed = 0
    for path in target.rglob("*.json"):
        path.unlink()
        removed += 1
    return removed
