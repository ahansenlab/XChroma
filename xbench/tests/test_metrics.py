"""Synthetic checks for the array-only metric package."""
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import pearsonr, spearmanr


def matrix(n=64):
    rng = np.random.default_rng(137)
    x = rng.normal(0.3, 0.5, (n, n))
    return np.clip((x + x.T) / 2, -2, 2)


class MetricChecks(unittest.TestCase):
    def test_all_nine_metrics_identity(self):
        from xbench import metrics
        self.assertEqual(len(metrics.all_names()), 9)
        x = matrix()
        for name in metrics.all_names():
            with self.subTest(metric=name):
                module = metrics.load(name)
                fn = module.make(5, 1)[1] if name == 'insulation' else module.score
                value = fn(x, x)
                self.assertTrue(np.isfinite(value), (name, value))
                self.assertAlmostEqual(value, 0 if name == 'mse' else 1, places=6)

    def test_paired_pixel_correlations(self):
        from xbench.metrics import pearson, spearman
        x, y = matrix(), matrix() + 0.1
        y[5, 20] = 1.5
        x[1, 3], y[2, 7] = np.nan, np.nan
        valid = np.isfinite(x) & np.isfinite(y)
        self.assertAlmostEqual(pearson.score(x, y), pearsonr(x[valid], y[valid]).statistic)
        self.assertAlmostEqual(spearman.score(x, y), spearmanr(x[valid], y[valid]).statistic)

    def test_ssim_gt_first_and_fine_structure(self):
        from xbench.metrics import ssim
        from xbench.metrics._pollard import pollard
        from xbench.metrics.fine_structure import dog
        gt, pred = matrix(), matrix() * 0.4 + 0.2
        pred[3, 11] = np.nan
        for a, b in ((pred, gt), (dog(pred, 1), dog(gt, 1))):
            self.assertAlmostEqual(ssim.score(a, b), pollard.ssim_map(b, a))

    def test_dog_constant_and_missing(self):
        from xbench.metrics.fine_structure import dog
        x = np.ones((64, 64)); x[6:9, 8:10] = np.nan; x[21, 25] = 0
        missing = ~np.isfinite(x) | (x == 0)
        for res_kb in (1, 2):
            out = dog(x, res_kb)
            self.assertTrue(np.isnan(out[missing]).all())
            np.testing.assert_allclose(out[~missing], 0, atol=1e-12)

    def test_multiscale_insulation(self):
        from xbench.metrics import insulation
        x = matrix(100)
        for scale in (50, 100, 150, 250):
            name, fn = insulation.make(scale // 10, 10)
            self.assertEqual(name, f'Insul{scale}kb')
            self.assertAlmostEqual(fn(x, x), 1)

    def test_insulation_exponentiates_log_oe(self):
        from cooltools.api.insulation import _insul_diamond_dense
        from xbench.metrics import insulation
        x = matrix(); y = x.copy(); y[11:20, 30:40] += 0.1
        a, b = [_insul_diamond_dense(np.exp(z), window=5, ignore_diags=2, norm_by_median=True) for z in (x,y)]
        keep = np.isfinite(a) & np.isfinite(b)
        self.assertAlmostEqual(insulation.make(5, 1)[1](x,y), spearmanr(a[keep],b[keep]).statistic)

    def test_focal_operator_and_boundaries(self):
        from xbench.metrics.loop_strength import focal_strength
        x = np.zeros((15, 15)); x[5:8, 8:11] = 2
        self.assertEqual(focal_strength(x, 6, 9), 2)
        self.assertTrue(np.isnan(focal_strength(x, 1, 9)))
        x[3:10, 6:13] = np.nan
        self.assertTrue(np.isnan(focal_strength(x, 6, 9)))

    def test_joint_loop_retention(self):
        from xbench.metrics.loop_strength import score_maps, score, correlations
        a = matrix(); b = a.copy(); b[7:14, 27:34] = np.nan
        ia, ib = np.array([10,20,30]), np.array([30,40,50])
        values = score_maps(a,a,b,ia,ib,1)
        np.testing.assert_array_equal(values['retained_index'], [1,2])
        self.assertEqual(values['gt'].dtype, np.float32)
        np.testing.assert_array_equal(values['gt'],values['ours'])
        self.assertEqual(correlations(values['gt'],values['ours'],values['opp'])['n'],2)
        self.assertAlmostEqual(score(a,a,ia,ib,1),1)
        self.assertTrue(np.isnan(score(a,a,np.array([],dtype=int),np.array([],dtype=int),1)))

    def test_loop_input_contract(self):
        from xbench.metrics.loop_strength import score_maps, correlations
        x = matrix(); ia,ib = np.array([10]),np.array([30])
        for args in ((x,x,x,ia,ib,10),(x,x,x,ib,ia,1),(x,x,x,np.array([10.]),ib,1),(x*100,x,x,ia,ib,1)):
            with self.assertRaises(ValueError): score_maps(*args)
        with self.assertRaises(ValueError): correlations([1,np.nan],[1,2],[1,2])

    def test_loop_correlation_paired_values(self):
        from xbench.metrics.loop_strength import correlations
        gt = np.array([.1,.2,.5,.3]); pred = np.array([.2,.25,.4,.27]); opp=gt[::-1]
        out = correlations(gt,pred,opp)
        self.assertAlmostEqual(out['XChroma_Pearson_r'],pearsonr(gt,pred).statistic)
        self.assertAlmostEqual(out['comparison_model_Pearson_r'],pearsonr(gt,opp).statistic)
        self.assertIsNone(correlations([1,1],[1,2],[2,3])['XChroma_Pearson_r'])

    def test_environment_cannot_enable_historical_zero_fill(self):
        from xbench.metrics import eigenvector, distance_enrichment, contact_directionality
        x,y=matrix(),matrix()*.8; x[4,9]=np.nan; y[5,11]=np.nan
        for module in (eigenvector,distance_enrichment,contact_directionality):
            before=module.score(x,y)
            with patch.dict(os.environ,{'KEEP_MISSING':'0','XCHROM_BENCH_LEGACY':'1'}):
                after=module.score(x,y)
            self.assertEqual(before,after)

    def test_registry_and_imports_are_standalone(self):
        from xbench import metrics
        with self.assertRaises(ValueError): metrics.load('pollard_insulation')
        with self.assertRaises(ValueError): metrics.load('missing')
        code='''import sys
from xbench import metrics
for name in metrics.all_names(): metrics.load(name)
from xbench.metrics import fine_structure, loop_strength
assert not any(n in sys.modules for n in ('xbench.config','xbench.models','xbench.provenance','xtrain'))
'''
        root=Path(__file__).resolve().parents[2]
        result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
