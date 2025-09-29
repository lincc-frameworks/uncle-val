import importlib


def import_by_name(full_name: str):
    """Imports and returns an object by its fully qualified name."""
    try:
        module_name, object_name = full_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, object_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not locate object '{full_name}'.") from e