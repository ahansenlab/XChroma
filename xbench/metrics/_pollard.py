"""Load the preserved Pollard sources by explicit file path, without model imports."""
from pathlib import Path
import hashlib
import importlib.util
import os
import sys

DEFAULT_CODE = Path(__file__).resolve().parent / 'vendor/pollard'


def source_directory():
    path = Path(os.environ.get('XCHROMA_POLLARD_CODE', os.environ.get('XCHROM_POLLARD_CODE', DEFAULT_CODE))).expanduser().resolve()
    for name in ('scoring.py', 'hicrep.py'):
        if not (path / name).is_file():
            raise ImportError(f'Missing Pollard source {path / name}; set XCHROMA_POLLARD_CODE to an explicit verified source directory.')
    return path


def source_info():
    path = source_directory()
    return {'directory': str(path), 'sha256': {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in ('scoring.py', 'hicrep.py')}}


class _Pollard:
    _module = None
    def _load(self):
        if self._module is not None: return self._module
        path = source_directory()
        def load(name, file):
            spec = importlib.util.spec_from_file_location(name, path / file)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module
        old = sys.modules.get('hicrep')
        try:
            sys.modules['hicrep'] = load('_xchroma_pollard_hicrep', 'hicrep.py')
            self._module = load('_xchroma_pollard_scoring', 'scoring.py')
        except ImportError as e:
            raise ImportError(f'Cannot load preserved Pollard scoring from {path}: {e}. Use the documented scoring environment with numpy/scipy/pandas/scikit-image/scikit-learn/matplotlib/cooltools; no substitute metric is used.') from e
        finally:
            if old is None: sys.modules.pop('hicrep', None)
            else: sys.modules['hicrep'] = old
        return self._module
    def __getattr__(self, name): return getattr(self._load(), name)


pollard = _Pollard()
