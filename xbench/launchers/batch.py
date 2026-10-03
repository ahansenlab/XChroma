"""Plan, run or submit isolated current benchmark jobs; merge only completed shards."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CELLS = ['GM12878', 'K562', 'HepG2', 'HCT116']


def check_batch_environment():
    if os.environ.get('EPCOT_DIR'):
        raise ValueError('Full batch includes three EPCOT heads: unset global EPCOT_DIR; each head uses its distinct configured store. Use a single-head scorer for an explicit directory override.')


def plan(out):
    check_batch_environment()
    out = Path(out).resolve(); jobs = []
    specs = [(m, m, '') for m in ('akita', 'epiphany', 'corigami', 'alphagenome')]
    specs += [(f'epcot_{h}_{c}', f'epcot:{h}', c) for h in ('microc', 'intacthic', 'rcmc') for c in CELLS]
    for key, line, cell in specs:
        env = {'OUT': str(out / 'parts' / key), 'LINES': line, 'ONLY_CELL': cell,
               'PART': key, 'PERWIN': '1', 'EPCOT_HEAD': line.partition(':')[2] or 'intacthic'}
        jobs.append({'key': key, 'env': env, 'command': [sys.executable, '-u', '-m', 'xbench.run_filtered']})
    return {'repo': str(ROOT), 'out': str(out), 'jobs': jobs,
            'run_launcher': str(ROOT / 'xbench/launchers/run_one.sbatch'),
            'merge_launcher': str(ROOT / 'xbench/launchers/agg_one.sbatch')}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_current():
    check_batch_environment()
    from xbench.config import legacy_mode, require_prediction_store
    if legacy_mode(): raise ValueError('Batch entry is current-model only; do not bypass guards with legacy mode.')
    return require_prediction_store()


def begin(out):
    check_current()
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError('Batch OUT must be a fresh directory.')
    out.mkdir(parents=True, exist_ok=True)
    p = plan(out)
    (out / 'plan.json').write_text(json.dumps(p, indent=2) + '\n')
    return p


def read_plan(out):
    p = json.loads((Path(out) / 'plan.json').read_text())
    if Path(p['out']).resolve() != Path(out).resolve() or Path(p['repo']).resolve() != ROOT:
        raise ValueError('Plan root does not match this checkout/output.')
    return p


def run_task(out, key):
    check_current(); p = read_plan(out)
    job = next((j for j in p['jobs'] if j['key'] == key), None)
    if job is None: raise ValueError('Unknown task key.')
    env = {**os.environ, **job['env'], 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    # The scorer enforces a fresh task OUT and writes its identity receipt.
    subprocess.run([sys.executable, '-u', '-m', 'xbench.run_filtered'], cwd=ROOT, env=env, check=True)
    folder = Path(job['env']['OUT'])
    names = [f'part_{key}.tsv', f'perwin_{key}.tsv', 'declared_prediction_identity.json', 'comparison_sources.json']
    for name in names[:2]:
        with (folder / name).open() as f:
            if sum(1 for _ in f) < 2: raise ValueError(f'Task {key} produced no scores in {name}.')
    receipt = {'task': key, 'files': {name: digest(folder / name) for name in names}}
    (folder / 'completed.json').write_text(json.dumps(receipt, indent=2) + '\n')


def merge(out):
    p = read_plan(out); shards = []; identity = None
    for j in p['jobs']:
        d = Path(j['env']['OUT']); key = j['key']
        completed = json.loads((d / 'completed.json').read_text())
        if completed['task'] != key: raise ValueError('Completion task mismatch.')
        for name in [f'part_{key}.tsv', f'perwin_{key}.tsv', 'declared_prediction_identity.json', 'comparison_sources.json']:
            if digest(d / name) != completed['files'].get(name): raise ValueError(f'Incomplete or changed shard: {key}/{name}')
        ident = json.loads((d / 'declared_prediction_identity.json').read_text())
        if identity is None: identity = ident
        elif ident != identity: raise ValueError('Shards declare different prediction identities.')
        sources = json.loads((d / 'comparison_sources.json').read_text())
        module = 'xbench.models.' + j['env']['LINES'].partition(':')[0]
        keys = [k for k, v in sources.items() if v['adapter'] == module]
        if len(keys) != 1: raise ValueError('Shard source receipt does not match planned adapter.')
        if module == 'xbench.models.epcot':
            head = j['env']['EPCOT_HEAD']
            if keys[0] != 'epcot_' + head or sources[keys[0]].get('adapter_settings', {}).get('_HEAD') != head:
                raise ValueError('EPCOT shard head differs from its plan.')
        j = {**j, 'result_key': keys[0], 'comparison_sources': sources}
        shards.append((j, d, completed))
    dest = Path(out) / 'merged'
    if dest.exists(): raise ValueError('Merged output already exists; it will not be overwritten.')
    dest.mkdir()
    counts = {}
    for kind, name in [('part', 'summary.tsv'), ('perwin', 'perwin.tsv')]:
        seen = set(); fields = None; count = 0
        with (dest / name).open('w', newline='') as target:
            writer = None
            for j, d, _ in shards:
                with (d / f'{kind}_{j["key"]}.tsv').open() as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    if fields is None:
                        fields = reader.fieldnames; writer = csv.DictWriter(target, fieldnames=fields, delimiter='\t', lineterminator='\n'); writer.writeheader()
                    elif fields != reader.fieldnames: raise ValueError('Shard header mismatch.')
                    expected_model = j['result_key']
                    for r in reader:
                        if r['model'] != expected_model or (j['env']['ONLY_CELL'] and r['cell'] != j['env']['ONLY_CELL']):
                            raise ValueError('Shard contains an unexpected model or cell.')
                        keys = ['model', 'cell', 'metric', 'filt'] + (['chrom', 'start', 'end'] if kind == 'perwin' else [])
                        key = tuple(r[k] for k in keys)
                        if key in seen: raise ValueError(f'Duplicate row: {key}')
                        seen.add(key); writer.writerow(r); count += 1
        counts[name] = count
    report = {'tasks': len(shards), 'rows': counts, 'prediction_identity': identity,
              'shards': [{**c, 'comparison_sources': j['comparison_sources']} for j, _, c in shards], 'files': {n: digest(dest / n) for n in counts}}
    (dest / 'merge_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['plan', 'serial', 'submit', 'task', 'merge'])
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--task')
    a = ap.parse_args(argv)
    if a.mode == 'plan': print(json.dumps(plan(a.out), indent=2)); return
    if a.mode == 'task': run_task(a.out, a.task); return
    if a.mode == 'merge': print(json.dumps(merge(a.out), indent=2)); return
    p = begin(a.out)
    if a.mode == 'serial':
        for j in p['jobs']: run_task(a.out, j['key'])
        print(json.dumps(merge(a.out), indent=2)); return
    ids = []
    for j in p['jobs']:
        env = {**os.environ, 'XCHROM_REPO_ROOT': str(ROOT), 'XBENCH_BATCH_ROOT': p['out'], 'XBENCH_TASK': j['key']}
        jobid = subprocess.check_output(['sbatch', '--parsable', '--export=ALL', '--job-name=xb_' + j['key'], p['run_launcher']], env=env, text=True).strip().split(';')[0]
        if not jobid.isdigit(): raise ValueError(f'Unexpected Slurm job id: {jobid}')
        ids.append(jobid)
        (Path(a.out) / 'submitted.json').write_text(json.dumps({'jobs': ids, 'merge_job': None}, indent=2) + '\n')
    env = {**os.environ, 'XCHROM_REPO_ROOT': str(ROOT), 'XBENCH_BATCH_ROOT': p['out']}
    agg = subprocess.check_output(['sbatch', '--parsable', '--export=ALL', '--dependency=afterok:' + ':'.join(ids), p['merge_launcher']], env=env, text=True).strip()
    (Path(a.out) / 'submitted.json').write_text(json.dumps({'jobs': ids, 'merge_job': agg}, indent=2) + '\n')
    print(json.dumps({'jobs': ids, 'merge_job': agg}))


if __name__ == '__main__': main()
