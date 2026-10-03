"""Explicit paths for prepared benchmark inputs."""
import os
from pathlib import Path


def require_directory(variable, env=None):
    env = os.environ if env is None else env
    value = env.get(variable)
    if not value:
        raise ValueError(f"Set {variable} to the prepared input directory; see xbench/README.md.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{variable} is not a directory: {path}")
    return str(path)


def epcot_directory(head, env=None):
    env = os.environ if env is None else env
    variable = 'EPCOT_DIR' if env.get('EPCOT_DIR') else f'EPCOT_{head.upper()}_DIR'
    return require_directory(variable, env)
