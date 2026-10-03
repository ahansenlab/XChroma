"""Synthetic numerical and input-contract checks; no genome data or weights."""
from pathlib import Path
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def matrix(n=64):
    rng = np.random.default_rng(137)
    x = rng.normal(0.3, 0.5, (n, n))
    return np.clip((x + x.T) / 2, -2, 2)


class ScoreChecks(unittest.TestCase):
    def test_all_metrics_identity(self):
        from xbench import metrics
        x = matrix()
        for name in metrics.all_names():
            with self.subTest(metric=name):
                mod = metrics.load(name)
                fn = mod.make(5, 1)[1] if name == 'insulation' else mod.score
                score = fn(x, x)
                self.assertTrue(np.isfinite(score), (name, score))
                self.assertAlmostEqual(score, 0.0 if name == 'mse' else 1.0, places=6)

    def test_correlations_use_paired_missing_pixels(self):
        from xbench.metrics import pearson, spearman
        x, y = matrix(), matrix() + 0.1
        y[5, 20] = 1.5
        x[1, 3], y[2, 7] = np.nan, np.nan
        valid = np.isfinite(x) & np.isfinite(y)
        self.assertAlmostEqual(pearson.score(x, y), pearsonr(x[valid], y[valid]).statistic)
        self.assertAlmostEqual(spearman.score(x, y), spearmanr(x[valid], y[valid]).statistic)

    def test_ssim_uses_gt_first(self):
        from xbench.metrics import ssim
        from xbench.metrics._pollard import pollard
        gt, pred = matrix(), matrix() * 0.4 + 0.2
        pred[3, 11] = np.nan
        self.assertAlmostEqual(ssim.score(pred, gt), pollard.ssim_map(gt, pred))

    def test_log_space_and_missing(self):
        from xbench.space import clip, snl, agg
        x = np.array([[2., 0., 4.], [0., 6., 8.], [4., 8., 10.]])
        y = snl(x)
        self.assertTrue(np.isnan(y[0, 1]))
        self.assertAlmostEqual(y[0, 0], np.log(2 / 6))
        self.assertAlmostEqual(y[1, 2], 0)
        np.testing.assert_allclose(agg(np.arange(16).reshape(4, 4), 2), [[2.5, 4.5], [10.5, 12.5]])
        with patch.dict(os.environ, {'KEEP_MISSING': '1'}):
            self.assertTrue(np.isnan(clip(np.array([np.nan, 3]))[0]))
            self.assertEqual(clip(np.array([3]))[0], 2)

    def test_dog_constant_and_missing_mask(self):
        from xbench.loop_strength import dog
        from xbench.run_filtered import dog as driver_dog, sigmas_for
        x = np.ones((64, 64)); x[6:9, 8:10] = np.nan; x[21, 25] = 0
        out = dog(x, 1)
        missing = ~np.isfinite(x) | (x == 0)
        self.assertTrue(np.isnan(out[missing]).all())
        np.testing.assert_allclose(out[~missing], 0, atol=1e-12)
        np.testing.assert_allclose(driver_dog(x), out, equal_nan=True)
        self.assertEqual(sigmas_for(2)[0], 1.6)

    def test_focal_operator_and_shared_retention(self):
        from xbench.loop_strength import focal_strength, score_maps
        x = np.zeros((15, 15)); x[5:8, 8:11] = 2
        self.assertEqual(focal_strength(x, 6, 9), 2)
        self.assertTrue(np.isnan(focal_strength(x, 1, 9)))
        a = matrix(64); b = a.copy(); b[7:14, 27:34] = np.nan
        values = score_maps(a, a, b, np.array([10, 20, 30]), np.array([30, 40, 50]), 1)
        np.testing.assert_array_equal(values['retained_index'], [1, 2])
        self.assertEqual(values['gt'].dtype, np.float32)
        np.testing.assert_array_equal(values['gt'], values['ours'])

    def test_loop_manifest_cli_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'window.npz'
            gt = matrix(64)
            np.savez_compressed(source, gt=gt, ours=gt, opp=gt * .9,
                                ia=np.array([10, 20, 30]), ib=np.array([30, 40, 50]))
            manifest = {'input_space': 'clipped_natural_log_observed_expected', 'windows': [
                {'npz': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'cell': 'example', 'line': 'example', 'res_kb': 1, 'chrom': 'chr1', 'window_start': 0}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            output = root / 'scores'
            subprocess.run([sys.executable, '-m', 'xbench.loop_strength', 'maps', '--manifest', str(root / 'manifest.json'), '--out', str(output)], cwd=ROOT, capture_output=True, text=True, check=True)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['rows'][0]['n'], 3)
            self.assertAlmostEqual(report['rows'][0]['XChrom_Pearson_r'], 1)
            replay = subprocess.run([sys.executable, '-m', 'xbench.loop_strength', 'replay', '--input-dir', str(output)], cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(replay.stdout)['groups'], 1)


class ContractChecks(unittest.TestCase):
    def test_explicit_directory_and_heads(self):
        from xbench.paths import require_directory, epcot_directory
        with self.assertRaises(ValueError): require_directory('GT_TILES', {})
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(epcot_directory('microc', {'EPCOT_MICROC_DIR': d}), str(Path(d).resolve()))

    def test_missing_values_guard(self):
        from xbench.config import keep_missing
        with self.assertRaises(ValueError): keep_missing({'KEEP_MISSING': '0'})
        self.assertTrue(keep_missing({}))

    def test_cache_identity_requires_all_members(self):
        from xbench.config import require_prediction_store
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            locked = json.loads((ROOT / 'xtrain/checkpoints/current_model.json').read_text())
            manifest = {k: locked[k] for k in ('release_id', 'members', 'ensemble')}
            manifest['ensemble'] = {'operator':'arithmetic_mean_outputs','space':'natural_log_observed_expected'}
            manifest['prediction_root'] = str(root)
            p = root / 'manifest.json'; p.write_text(json.dumps(manifest))
            env = {'OURS_PRED_BASE': str(root), 'XCHROM_CACHE_MANIFEST': str(p)}
            self.assertEqual(require_prediction_store(env), str(root.resolve()))
            manifest['members'] = manifest['members'][:-1]; p.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): require_prediction_store(env)

    def test_batch_plan_independent_heads(self):
        from xbench.launchers.batch import plan
        with patch.dict(os.environ, {}, clear=True):
            p = plan('/tmp/xchrom-example-output')
            self.assertEqual(len(p['jobs']), 16)
            self.assertEqual(len({j['env']['OUT'] for j in p['jobs']}), 16)
            self.assertEqual({j['env']['EPCOT_HEAD'] for j in p['jobs'] if j['key'].startswith('epcot_')}, {'microc', 'intacthic', 'rcmc'})

    def test_help_without_external_inputs(self):
        for module in ('xbench.run', 'xbench.run_filtered', 'xbench.loop_strength', 'xbench.comparators.corigami.predict', 'xbench.comparators.epiphany.fetch_bam_urls'):
            with self.subTest(module=module):
                env = {k: v for k, v in os.environ.items() if k not in ('OURS_PRED_BASE', 'OURS_BASE', 'XCHROM_CACHE_MANIFEST', 'GT_TILES', 'OUT')}
                result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_epcot_cached_driver_and_alignment(self):
        from xbench.config import RELEASE_ID
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cell = 'GM12878'
            for directory in ('ours', 'gt', 'epcot'): (root / directory / cell).mkdir(parents=True)
            starts = [0]
            for path in (root / 'ours' / cell / 'chr1.pred.manifest.json', root / 'gt' / cell / 'chr1.w1000k.s1000k.manifest.json'):
                path.write_text(json.dumps({'starts': starts}))
            rng = np.random.default_rng(24)
            x = rng.uniform(.2, 1.8, size=(1000,1000)).astype(np.float32); x=(x+x.T)/2
            for path, values in (
                (root / 'ours' / cell / 'chr1.pred.f16.memmap', np.log(x)),
                (root / 'epcot' / cell / 'chr1.epcot.f16.memmap', np.log(x)/np.log(2)),
                (root / 'gt' / cell / 'chr1.w1000k.s1000k.f16.memmap', x)):
                array = np.memmap(path, dtype=np.float16, mode='w+', shape=(1,1000,1000)); array[0]=values; array.flush(); del array
            locked = json.loads((ROOT / 'xtrain/checkpoints/current_model.json').read_text())
            manifest = {k: locked[k] for k in ('release_id','members','ensemble')}; manifest['prediction_root']=str(root/'ours'); manifest['ensemble']={'operator':'arithmetic_mean_outputs','space':'natural_log_observed_expected'}
            cache = root / 'cache.json'; cache.write_text(json.dumps(manifest))
            env = {**os.environ, 'OURS_PRED_BASE':str(root/'ours'), 'XCHROM_CACHE_MANIFEST':str(cache),
                   'GT_TILES':str(root/'gt'), 'EPCOT_INTACTHIC_DIR':str(root/'epcot'), 'EPCOT_HEAD':'intacthic',
                   'OUT':str(root/'out'), 'LINES':'epcot:intacthic', 'ONLY_CELL':cell, 'ONLY_METRIC':'pearson,spearman,ssim', 'LIMIT':'1', 'PERWIN':'1'}
            env.pop('EPCOT_DIR',None)
            result = subprocess.run([sys.executable, '-m', 'xbench.run_filtered'], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with (root/'out/summary.tsv').open() as stream: rows=list(csv.DictReader(stream,delimiter='\t'))
            self.assertEqual(len(rows),6)
            self.assertEqual({r['filt'] for r in rows},{'raw','dog'})
            self.assertTrue(all(r['n']=='1' and np.isfinite(float(r['XChrom'])) for r in rows))
            for row in rows: self.assertAlmostEqual(float(row['XChrom']),float(row['opp']),delta=.002)
            with patch.dict(os.environ,env):
                mod=importlib.import_module('xbench.models.epcot');mod=importlib.reload(mod)
                (root/'ours'/cell/'chr1.pred.manifest.json').write_text(json.dumps({'starts':[1000000]}))
                with self.assertRaisesRegex(ValueError,'identical window starts/order'): mod.load_ours(cell,'chr1',0)


if __name__ == '__main__': unittest.main()
