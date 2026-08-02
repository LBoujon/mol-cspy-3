"""MPI orchestration classes, imported lazily to avoid initializing MPI early."""

from importlib import import_module


__all__ = [
    "CSPyTaskTag",
    "CSPyWorker",
    "Master",
    "ReoptTaskTag",
    "ReoptWorker",
    "WorkQueue",
    "Worker",
]

_EXPORTS = {
    "Master": ("cspy.cspympi.master", "Master"),
    "WorkQueue": ("cspy.cspympi.work_queue", "WorkQueue"),
    "Worker": ("cspy.cspympi.worker", "Worker"),
    "CSPyWorker": ("cspy.cspympi.cspy_tasks", "CSPyWorker"),
    "CSPyTaskTag": ("cspy.cspympi.cspy_tasks", "CSPyTaskTag"),
    "ReoptWorker": ("cspy.cspympi.reopt_tasks", "ReoptWorker"),
    "ReoptTaskTag": ("cspy.cspympi.reopt_tasks", "ReoptTaskTag"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
