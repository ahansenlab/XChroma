"""GT-defined loop strengths on DoG maps and Pearson agreement.

Score prepared maps or replay position-paired focal-strength vectors.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr


def dog(matrix, res_kb):
    """Preserve the producer's nonzero-finite mask, physical widths and boundary."""
    x = np.asarray(matrix, np.float64)
    m = np.isfinite(x) & (x != 0)
    mask = m.astype(np.float64)
    xf = np.where(mask > 0, x, 0.0)
    s1 = 3.2 / float(res_kb)
    def smooth(s):
        den = gaussian_filter(mask, s, mode='nearest')
        return np.where(den > 1e-6,
                        gaussian_filter(xf, s, mode='nearest') / np.maximum(den, 1e-6), np.nan)
    return np.where(m, smooth(s1) - smooth(s1 * 2 ** (1.0 / 10)), np.nan)


def focal_strength(filtered, ia, ib):
    """Central 3x3 minus four 7-bin edges; corners occur twice, as in variant G."""
    n = filtered.shape[0]
    if ia - 3 < 0 or ia + 3 >= n or ib - 3 < 0 or ib + 3 >= n:
        return np.nan
    patch = filtered[ia - 3:ia + 4, ib - 3:ib + 4]
    if np.isfinite(patch).mean() < 0.8:
        return np.nan
    border = np.concatenate([patch[0].ravel(), patch[-1].ravel(),
                             patch[:, 0].ravel(), patch[:, -1].ravel()])
    return float(np.nanmean(patch[2:5, 2:5]) - np.nanmean(border))


def score_maps(gt, ours, opp, ia, ib, res_kb):
    """Input is already comparison-grid ln(O/E), clipped to [-2,2]."""
    maps = [np.asarray(x, np.float64) for x in (gt, ours, opp)]
    if any(x.ndim != 2 or x.shape != maps[0].shape or x.shape[0] != x.shape[1] for x in maps):
        raise ValueError('Expected three square maps on one scoring grid.')
    if float(res_kb) not in (1.0, 2.0):
        raise ValueError('Loop-scoring grids must be 1 or 2 kb.')
    if any(np.isinf(x).any() or (np.abs(x[np.isfinite(x)]) > 2).any() for x in maps):
        raise ValueError('Inputs must already be clipped ln(O/E); infinities are invalid.')
    ia, ib = np.asarray(ia), np.asarray(ib)
    if ia.ndim != 1 or ib.shape != ia.shape or not np.issubdtype(ia.dtype, np.integer) or not np.issubdtype(ib.dtype, np.integer):
        raise ValueError('ia and ib must be matching integer vectors of GT-defined sites.')
    filtered = [dog(x, res_kb) for x in maps]
    retained, values = [], []
    for k, (a, b) in enumerate(zip(ia, ib)):
        if a >= b:
            raise ValueError('GT positions must be in the upper triangle (ia < ib).')
        v = [focal_strength(x, int(a), int(b)) for x in filtered]
        if np.isfinite(v).all():
            retained.append(k)
            values.append(v)
    # Producer exports scalar vectors as float32 before Pearson replay.
    a = np.asarray(values, np.float32).reshape(-1, 3)
    return {'gt': a[:, 0], 'ours': a[:, 1], 'opp': a[:, 2],
            'ia': ia[retained], 'ib': ib[retained], 'retained_index': np.asarray(retained, np.int64)}


def correlations(gt, ours, opp):
    a = [np.asarray(x, np.float64) for x in (gt, ours, opp)]
    if any(x.ndim != 1 or x.shape != a[0].shape for x in a):
        raise ValueError('Scalar vectors must be one-dimensional and position-paired.')
    if not all(np.isfinite(x).all() for x in a):
        raise ValueError('Delivered scalar vectors must contain only jointly retained finite sites.')
    def corr(x):
        return float(pearsonr(a[0], x).statistic) if len(x) >= 2 and np.ptp(a[0]) > 0 and np.ptp(x) > 0 else None
    return {'n': len(a[0]), 'XChrom_Pearson_r': corr(a[1]), 'comparison_model_Pearson_r': corr(a[2])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output must be an empty directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def replay(input_dir, table=None):
    names = {'Akita': 'akita', 'AlphaGenome': 'alphagenome', 'EPCOT intact Hi-C': 'epcot_intacthic',
             'EPCOT Micro-C': 'epcot_microc', 'EPCOT RCMC': 'epcot_rcmc'}
    expected = {}
    if table:
        with Path(table).open() as f:
            expected = {(r['cell_type'], names[r['comparison_configuration']]): r for r in csv.DictReader(f, delimiter='\t')}
    rows = []
    for p in sorted(Path(input_dir).glob('*.npz')):
        with np.load(p, allow_pickle=False) as z:
            cell, line = str(z['cell'].item()), str(z['line'].item())
            row = {'cell': cell, 'line': line, **correlations(z['gt'], z['ours'], z['opp']), 'sha256': sha(p)}
        if table:
            e = expected.pop((cell, line))
            assert row['n'] == int(e['scored_reference_loops'])
            for k in ('XChrom_Pearson_r', 'comparison_model_Pearson_r'):
                diff = abs(row[k] - float(e[k])); assert diff < 1e-8, (cell, line, k, diff)
                row[k + '_abs_difference'] = diff
        rows.append(row)
    if not rows or expected:
        raise ValueError('Missing scalar exports or unmatched table groups.')
    return {'scope': 'delivered_scalar_replay', 'groups': len(rows), 'correlations': 2 * len(rows), 'rows': rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('replay', help='Recompute Pearson from delivered scalar NPZ files.')
    r.add_argument('--input-dir', type=Path, required=True); r.add_argument('--table', type=Path)
    r.add_argument('--out', type=Path, help='Optional fresh report directory; otherwise JSON to stdout.')
    m = sub.add_parser('maps', help='Extract strengths from explicit prepared maps and GT sites.')
    m.add_argument('--manifest', type=Path, required=True); m.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.mode == 'replay':
        report = replay(a.input_dir, a.table)
        if a.out: (fresh(a.out) / 'replay.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        manifest = json.loads(a.manifest.read_text())
        if manifest.get('input_space') != 'clipped_natural_log_observed_expected':
            raise ValueError('Manifest must declare prepared clipped_natural_log_observed_expected maps.')
        items = manifest['windows']; groups = {}; inputs = []
        if not items: raise ValueError('Manifest has no windows.')
        seen = set()
        for window_index, item in enumerate(items):
            path = (a.manifest.parent / item['npz']).resolve()
            key_path = (item['cell'], item['line'], str(path))
            if key_path in seen: raise ValueError('Duplicate input window within one comparison group.')
            seen.add(key_path)
            if item.get('sha256') != sha(path): raise ValueError(f'Input SHA256 mismatch: {path}')
            with np.load(path, allow_pickle=False) as z:
                values = score_maps(z['gt'], z['ours'], z['opp'], z['ia'], z['ib'], item['res_kb'])
            key = (item['cell'], item['line'])
            values['window_index'] = np.full(len(values['gt']), window_index, np.int64)
            groups.setdefault(key, []).append(values)
            inputs.append({**item, 'retained_loops': len(values['gt'])})
        out = fresh(a.out); rows = []
        for i, ((cell, line), values) in enumerate(sorted(groups.items())):
            vectors = {k: np.concatenate([v[k] for v in values]) for k in ('gt', 'ours', 'opp')}
            coordinates = {k: np.concatenate([v[k] for v in values]) for k in ('ia', 'ib', 'retained_index', 'window_index')}
            np.savez_compressed(out / f'group_{i:02d}.npz', **vectors, **coordinates, cell=cell, line=line)
            rows.append({'cell': cell, 'line': line, **correlations(**vectors)})
        report = {'scope': 'prepared_map_scoring', 'manifest_sha256': sha(a.manifest), 'inputs': inputs, 'rows': rows}
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
