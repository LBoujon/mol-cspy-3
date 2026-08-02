"""Distributed workflow managers, imported lazily to avoid early MPI startup."""

from importlib import import_module


__all__ = ["AUTCSP", "QuasiRandomCSP"]

_EXPORTS = {
    "QuasiRandomCSP": ("cspy.distributed.csp_manager", "QuasiRandomCSP"),
    "AUTCSP": ("cspy.distributed.aut_manager", "AUTCSP"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
