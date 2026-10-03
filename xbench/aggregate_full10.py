"""Summarize explicitly supplied current xbench summary TSVs by metric and cell.

The historical 24k aggregator is preserved under the repair audit's before/ tree.
Metrics are never averaged together. No prediction or scoring is performed.
"""
import argparse
import csv
import math
import re
import sys


def base_metric(name):
    return re.sub(r"__offdiag[01]$", "", name).upper()


def summarize(paths):
    groups, seen = {}, set()
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                model, cell, metric = r['model'], r['cell'], r['metric']
                filt = r.get('filt', 'raw')
                if filt not in ('raw', 'dog'): raise ValueError(f'Unknown filter: {filt}')
                if filt == 'dog' and base_metric(metric) == 'MSE': raise ValueError('MSE is whole-map only.')
                key = (model, cell, metric, filt)
                if key in seen: raise ValueError(f'Duplicate configuration/cell/metric/filter: {key}')
                seen.add(key)
                x, y = float(r['XChrom']), float(r['opp'])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f'Summary must contain paired finite cell means: {key}')
                groups.setdefault((model, metric, filt), []).append((x, y))
    if not seen: raise ValueError('No summary rows found.')
    rows = []
    for (model, metric, filt), vals in sorted(groups.items()):
        x, y = (sum(v[i] for v in vals) / len(vals) for i in (0, 1))
        lower = base_metric(metric) == 'MSE'
        rows.append({'model': model, 'metric': metric, 'maps': 'whole map' if filt == 'raw' else 'fine structure',
                     'cells': len(vals), 'XChrom_mean': x, 'comparison_mean': y,
                     'ratio': x / y if y > 0 else '',
                     'cells_won': sum(a < b if lower else a > b for a, b in vals)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', nargs='+', required=True, help='Explicit run_filtered summary/part TSVs (or whole-map run.py summaries).')
    args = p.parse_args(argv); rows = summarize(args.summary)
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    w.writeheader(); w.writerows(rows)


if __name__ == '__main__': main()
