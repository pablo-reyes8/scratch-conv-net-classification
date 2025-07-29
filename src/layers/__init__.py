import pkgutil
import importlib
import os


__all__ = [
    name
    for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)])
    if name != "__init__"]


for module_name in __all__:
    importlib.import_module(f"{__name__}.{module_name}")

