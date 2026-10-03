"""Record the actual adapter selection alongside each score shard."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys


def record_adapter(module, out, *, mode):
    from .metrics._pollard import source_info
    path = Path(module.__file__).resolve()
    key = getattr(module, 'KEY', module.__name__.rsplit('.', 1)[-1])
    target = Path(out) / 'comparison_sources.json'
    previous = json.loads(target.read_text()) if target.exists() else {}
    packages = {}
    for name in ('numpy', 'scipy', 'pandas', 'scikit-image', 'scikit-learn', 'cooltools'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    entry = {'adapter': module.__name__, 'module_path': str(path),
             'module_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'key': key, 'label': getattr(module, 'LABEL', key),
             'resolution_kb': getattr(module, 'RES_KB', None),
             'default_offdiag': bool(getattr(module, 'OFFDIAG', False)),
             'source_identity': getattr(module, 'SOURCE_IDENTITY', None),
             'configured_paths': {k: getattr(module, k) for k in ('STORE', 'GTDIR', '_OURS_BASE', '_OURS_PRED', '_EPD', '_GTD') if hasattr(module, k)},
             'adapter_settings': {k: getattr(module, k) for k in ('_HEAD', '_CONV', 'NB', 'BS') if hasattr(module, k)},
             'mode': mode, 'pollard_source': source_info(),
             'python': sys.executable, 'packages': packages,
             'options': {k: os.environ.get(k) for k in ('KEEP_MISSING', 'OFFDIAG', 'INSUL_BINS', 'INSUL_MAX_KB', 'SIGMA1_KB', 'DOG_LINES', 'ONLY_METRIC', 'METRICS', 'CELLS', 'ONLY_CELL', 'ONLY_CHROM', 'LINES', 'LIMIT')}}
    if key in previous and previous[key] != entry:
        raise ValueError(f'Adapter identity changed within output directory: {key}')
    previous[key] = entry
    target.write_text(json.dumps(previous, indent=2, default=str) + '\n')
    return entry
