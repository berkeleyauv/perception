from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pkgutil import walk_packages
from typing import Dict, Iterable, Tuple, Type


@dataclass(frozen=True)
class PerceiverKey:
    task: str
    algo: str


_REGISTRY: Dict[PerceiverKey, Type] = {}
_DISCOVERED_TASKS = set()


def register_perceiver(task: str, algo: str):
    """Register a TaskPerceiver subclass for CLI/runtime lookup."""

    def decorator(cls):
        cls.task = task
        cls.algo = algo
        _REGISTRY[PerceiverKey(task, algo)] = cls
        return cls

    return decorator


def discover_perceivers(task: str | None = None) -> None:
    """Import non-archived task modules so decorators can register classes."""
    discovery_key = task or "*"
    if discovery_key in _DISCOVERED_TASKS:
        return

    tasks_pkg = import_module("perception.tasks")
    for module_info in walk_packages(tasks_pkg.__path__, tasks_pkg.__name__ + "."):
        parts = module_info.name.split(".")
        if "_archive" in parts or module_info.ispkg:
            continue
        if "modules" in parts or "tests" in parts or "yolo" in parts:
            continue
        if task and len(parts) > 2 and parts[2] not in {task, "_examples"}:
            continue
        import_module(module_info.name)

    _DISCOVERED_TASKS.add(discovery_key)


def get_perceiver(task: str, algo: str):
    discover_perceivers(task)
    key = PerceiverKey(task, algo)
    if key not in _REGISTRY:
        available = ", ".join(
            f"{item.task}/{item.algo}"
            for item in sorted(_REGISTRY, key=lambda item: (item.task, item.algo))
        )
        raise KeyError(f"Unknown perceiver {task}/{algo}. Available: {available}")
    return _REGISTRY[key]


def list_perceivers() -> Iterable[Tuple[str, str]]:
    discover_perceivers()
    return sorted((key.task, key.algo) for key in _REGISTRY)
