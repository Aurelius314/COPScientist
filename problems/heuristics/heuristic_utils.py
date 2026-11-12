"""Small helpers shared by the migrated heuristic evaluators."""

import inspect


def get_heuristic_name(module, possible_names: list[str]) -> str | None:
    for name in possible_names:
        if hasattr(module, name) and inspect.isfunction(getattr(module, name)):
            return name
    return None
