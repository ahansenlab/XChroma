"""Configuration guards for future scoring; no model loading or inference.

A cache manifest is a producer declaration bound to a directory. These guards
validate that declaration, not the numerical contents or generating execution.
"""
from pathlib import Path
import json
import os
import re

RELEASE_ID = 'xchrom_avg10_present_30500_35000'
STEPS = list(range(30500, 35001, 500))


def legacy_mode(env=None):
    env = os.environ if env is None else env
    value = env.get('XCHROM_BENCH_LEGACY', '0')
    if value not in ('0', '1'):
        raise ValueError('XCHROM_BENCH_LEGACY must be 0 or 1.')
    return value == '1'


def keep_missing(env=None):
    env = os.environ if env is None else env
    value = env.get('KEEP_MISSING', '1')
    if value not in ('0', '1'):
        raise ValueError('KEEP_MISSING must be 0 or 1.')
    if value == '0' and not legacy_mode(env):
        raise ValueError('Current paper scoring requires KEEP_MISSING=1. Zero fill is available only with explicit XCHROM_BENCH_LEGACY=1.')
    return value == '1'


def configure_protocol():
    os.environ.setdefault('KEEP_MISSING', '1')
    keep_missing()


def selected_epcot_head(env=None):
    env = os.environ if env is None else env
    head = env.get('EPCOT_HEAD')
    if head is None and legacy_mode(env):
        head = 'microc'
    if head not in ('intacthic', 'microc', 'rcmc'):
        raise ValueError('Set EPCOT_HEAD explicitly: intacthic for Fig. 1b, or the intended supplementary head (microc/rcmc).')
    return head


def require_prediction_store(env=None, locked_path=None):
    env = os.environ if env is None else env
    keep_missing(env)
    roots = [Path(env[k]).expanduser().resolve() for k in ('OURS_PRED_BASE', 'OURS_BASE') if env.get(k)]
    if not roots:
        raise ValueError('Set OURS_PRED_BASE or OURS_BASE explicitly; no historical prediction store is selected.')
    if any(x != roots[0] for x in roots):
        raise ValueError('OURS_PRED_BASE and OURS_BASE point to different prediction stores.')
    root = roots[0]
    if not root.is_dir():
        raise ValueError(f'Prediction store does not exist: {root}')
    if legacy_mode(env):
        return str(root)
    manifest_name = env.get('XCHROM_CACHE_MANIFEST')
    if not manifest_name:
        raise ValueError('Current scoring requires XCHROM_CACHE_MANIFEST with the producer-declared directory and all ten checkpoint identities; see xbench/README.md.')
    manifest_path = Path(manifest_name).expanduser().resolve()
    data = json.loads(manifest_path.read_text())
    declared_root = Path(data['prediction_root']).expanduser()
    if not declared_root.is_absolute():
        declared_root = manifest_path.parent / declared_root
    if declared_root.resolve() != root:
        raise ValueError('Cache manifest prediction_root does not match the configured store.')
    locked_path = Path(locked_path) if locked_path is not None else Path(__file__).resolve().parents[1] / 'xtrain/checkpoints/current_model.json'
    locked = json.loads(locked_path.read_text())
    if data['release_id'] != RELEASE_ID or locked['release_id'] != RELEASE_ID:
        raise ValueError('Cache manifest is not the current locked-ten model.')
    def signature(rows):
        if len(rows) != 10:
            raise ValueError('Every one of the ten members is required.')
        result = [(int(x['step']), x['sha256'], float(x['weight'])) for x in rows]
        if [x[0] for x in result] != STEPS or any(x[2] != 0.1 or not re.fullmatch('[0-9a-f]{64}', x[1]) for x in result):
            raise ValueError('Cache manifest must list the exact ten steps in order, SHA256 values, and weight 0.1 each.')
        return result
    if signature(data['members']) != signature(locked['members']):
        raise ValueError('Cache member identities differ from current_model.json.')
    if data['ensemble'] != {'operator': 'arithmetic_mean_outputs', 'space': 'natural_log_observed_expected'}:
        raise ValueError('Cache must declare equal output averaging in native natural-log observed/expected space.')
    return str(root)


def prepare_output_directory(env=None, locked_path=None):
    env = os.environ if env is None else env
    root = require_prediction_store(env, locked_path)
    if legacy_mode(env):
        return  # The explicit historical wrapper retains its previous output behavior.
    name = env.get('OUT')
    if not name:
        raise ValueError('Set OUT to a fresh directory; current scores must not reuse historical per-window files.')
    path = Path(name).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Current scoring requires an empty OUT directory. Existing score files are not keyed by complete source identity; use a new directory.')
    path.mkdir(parents=True, exist_ok=True)
    (path / 'declared_prediction_identity.json').write_text(json.dumps({
        'prediction_root': root,
        'cache_manifest': str(Path(env['XCHROM_CACHE_MANIFEST']).expanduser().resolve()),
        'release_id': RELEASE_ID,
        'keep_missing': True,
        'scope': 'Configuration declaration verified; this does not independently verify the cache-generating inference.'
    }, indent=2) + '\n')
