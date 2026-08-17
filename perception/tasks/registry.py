"""Decorator-based registration for TaskPerceiver algorithms.

Replaces a hand-maintained ALGOS dict: decorate a TaskPerceiver subclass with
@register_perceiver(task=..., algo=...) and it becomes discoverable through
get_perceiver()/list_algos()/list_tasks() once discover_all() has imported it.

discover_all() walks perception.tasks and imports every module so the
decorators actually run — without it, an otherwise-correct @register_perceiver
never fires, since nothing else references the module. Modules that fail to
import (e.g. yolo/orientation code needing torch/ultralytics, which aren't
part of the classical-only dependency group) are skipped rather than raised,
so discovery works in an environment that only installs
requirements-classical.txt.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

from perception.tasks.TaskPerceiver import TaskPerceiver

_REGISTRY: dict[tuple[str, str], type[TaskPerceiver]] = {}
_DEFAULTS: dict[str, str] = {}


def register_perceiver(task: str, algo: str, default: bool = False):
    """Class decorator: register a TaskPerceiver subclass under (task, algo).

    Pass default=True to make this the algo get_default_algo() returns for
    `task` when a caller (e.g. the vis CLI) doesn't specify one explicitly.
    """

    def decorator(cls: type[TaskPerceiver]) -> type[TaskPerceiver]:
        key = (task, algo)
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"task={task!r} algo={algo!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}, cannot also "
                f"register {cls.__module__}.{cls.__qualname__}"
            )
        _REGISTRY[key] = cls
        if default:
            existing_default = _DEFAULTS.get(task)
            if existing_default is not None and existing_default != algo:
                raise ValueError(
                    f"task={task!r} already has default algo {existing_default!r}, "
                    f"cannot also mark {algo!r} as default"
                )
            _DEFAULTS[task] = algo
        return cls

    return decorator


def get_perceiver(task: str, algo: str) -> type[TaskPerceiver]:
    """Look up the TaskPerceiver subclass registered for (task, algo)."""
    try:
        return _REGISTRY[(task, algo)]
    except KeyError:
        raise KeyError(
            f"no perceiver registered for task={task!r} algo={algo!r}"
        ) from None


def list_tasks() -> list[str]:
    """All task names with at least one registered algo."""
    return sorted({task for task, _algo in _REGISTRY})


def list_algos(task: str) -> list[str]:
    """All algo names registered for a given task."""
    return sorted(algo for t, algo in _REGISTRY if t == task)


def get_default_algo(task: str) -> str:
    """The default algo for a task: whichever was marked default=True, or the
    sole registered algo if the task only has one. Raises KeyError if neither
    applies, i.e. the caller must specify an algo explicitly.
    """
    explicit = _DEFAULTS.get(task)
    if explicit is not None:
        return explicit
    algos = list_algos(task)
    if len(algos) == 1:
        return algos[0]
    raise KeyError(
        f"no default algo for task={task!r} (algos: {', '.join(algos) or 'none registered'})"
    )


_EXCLUDED_PACKAGES = ("perception.tasks._archive",)


def discover_all(package: str = "perception.tasks") -> None:
    """Import every module under `package` so @register_perceiver decorators run.

    Safe to call more than once (re-importing an already-imported module is a
    cheap no-op). `_archive/` is skipped outright — retired tasks aren't meant
    to join the registry, and some are old scripts with unguarded top-level
    code (argument parsing, `sys.exit`, ...) that isn't safe to import at all.
    Any other import-time failure (missing optional dependency, or unguarded
    side-effecting code elsewhere) is caught and skipped with a warning rather
    than allowed to abort discovery of everything else.
    """
    pkg = importlib.import_module(package)
    for module_info in pkgutil.walk_packages(
        pkg.__path__, prefix=f"{package}.", onerror=lambda _name: None
    ):
        if any(
            module_info.name == excluded or module_info.name.startswith(excluded + ".")
            for excluded in _EXCLUDED_PACKAGES
        ):
            continue
        try:
            importlib.import_module(module_info.name)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            warnings.warn(
                f"registry.discover_all: skipping {module_info.name}: {exc!r}"
            )
