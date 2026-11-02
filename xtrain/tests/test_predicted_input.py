"""Validate the xatac-to-XChroma boundary without weights or model inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xtrain.inputs import encode_dna, prepare_input, prepare_predicted_input, sha256
from xtrain.predict import main


class PredictedInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.fasta = self.directory / 'genome.fa'
        self.sequence = b'ACGTNACGTNACGTNA'
        self.fasta.write_bytes(b'>chr1\n' + self.sequence + b'\n')
        Path(str(self.fasta) + '.fai').write_text('chr1\t16\t6\t16\t17\n')
        self.path = self.directory / 'atac.npz'
        self.signal = np.arange(16, dtype=np.float32)
        self.normalized = (np.log1p(self.signal.astype(np.float64) * 2) / 3).astype(np.float32)
        self.metadata = dict(status='PASS', assembly='GRCh38', chrom='chr1', start=0, end=16,
                             coordinate_convention='0-based half-open', resolution_bp=1,
                             fasta_sha256=sha256(self.fasta), members=[],
                             xchroma_conversion=dict(formula='log1p(signal * fixed_training_scale) / gp99',
                                                     fixed_training_scale=2, gp99=3))
        self.write_arrays()

    def write_metadata(self):
        Path(str(self.path) + '.json').write_text(json.dumps(self.metadata))

    def write_arrays(self, legacy=False, **changes):
        arrays = dict(signal=self.signal, xchroma_input=self.normalized, valid=np.ones(16, dtype=bool),
                      start=np.int64(0), end=np.int64(16))
        if legacy:
            arrays['xchrom_input'] = arrays.pop('xchroma_input')
        arrays.update(changes)
        np.savez_compressed(self.path, **arrays)
        self.metadata['output_sha256'] = sha256(self.path)
        self.write_metadata()

    def prepare(self, **changes):
        args = dict(fasta=self.fasta, atac_npz=self.path, assembly='hg38', chrom='chr1', start=0, end=16)
        args.update(changes)
        return prepare_predicted_input(**args)

    def test_normalized_channel_is_exact_and_dna_is_unchanged(self):
        values, receipt = self.prepare()
        np.testing.assert_array_equal(values[:4], encode_dna(self.sequence))
        np.testing.assert_array_equal(values[4], self.normalized)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(receipt['unknown_dna_bases'], 3)
        self.assertIn('no additional transform', receipt['normalization'])
        self.assertIn('xchroma_conversion', receipt)
        self.assertNotIn('xchrom_conversion', receipt)

    def test_legacy_names_produce_identical_input_and_canonical_receipt(self):
        current, current_receipt = self.prepare()
        self.metadata['xchrom_conversion'] = self.metadata.pop('xchroma_conversion')
        self.write_arrays(legacy=True)
        legacy, receipt = self.prepare()
        np.testing.assert_array_equal(legacy, current)
        self.assertEqual(receipt['input_sha256'], current_receipt['input_sha256'])
        self.assertEqual(receipt['xchroma_conversion'], current_receipt['xchroma_conversion'])
        self.assertNotIn('xchrom_conversion', receipt)
        self.assertTrue(receipt['normalization'].startswith('xchroma_input '))

    def test_matching_new_and_legacy_aliases_are_accepted(self):
        self.metadata['xchrom_conversion'] = self.metadata['xchroma_conversion'].copy()
        self.write_arrays(xchrom_input=self.normalized.copy())
        values, _ = self.prepare()
        np.testing.assert_array_equal(values[4], self.normalized)

    def test_conflicting_new_and_legacy_aliases_are_rejected(self):
        self.metadata['xchrom_conversion'] = {**self.metadata['xchroma_conversion'], 'gp99': 4}
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Conflicting xchroma_conversion'):
            self.prepare()
        self.metadata['xchrom_conversion'] = self.metadata['xchroma_conversion'].copy()
        for legacy in (self.normalized + 1, self.normalized[:-1], self.normalized.astype(np.float64)):
            with self.subTest(dtype=legacy.dtype, shape=legacy.shape):
                self.write_arrays(xchrom_input=legacy)
                with self.assertRaisesRegex(ValueError, 'Conflicting xchroma_input'):
                    self.prepare()

    def test_legacy_inputs_retain_dtype_and_conversion_checks(self):
        self.metadata['xchrom_conversion'] = self.metadata.pop('xchroma_conversion')
        for legacy, message in ((self.normalized.astype(np.float64), 'float32'),
                                (np.log1p(self.normalized).astype(np.float32), 'disagrees')):
            with self.subTest(message=message):
                self.write_arrays(legacy=True, xchrom_input=legacy)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()

    def test_coordinate_assembly_and_resolution_mismatch_rejected(self):
        for changed in ({'assembly': 'mm39'}, {'chrom': 'chr2'}, {'start': 1}, {'end': 15}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.prepare(**changed)
        self.metadata['resolution_bp'] = 50
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'resolution_bp'):
            self.prepare()

    def test_changed_npz_and_fasta_rejected(self):
        self.path.write_bytes(self.path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.prepare()
        self.write_arrays()
        self.fasta.write_bytes(b'>chr1\n' + b'T' * 16 + b'\n')
        with self.assertRaisesRegex(ValueError, 'FASTA differs'):
            self.prepare()

    def test_wrong_shape_or_missing_bases_rejected(self):
        for changes in ({'xchroma_input': self.normalized[:-1]}, {'valid': np.zeros(16, dtype=bool)},
                        {'xchroma_input': self.normalized.astype(np.float64)}, {'start': np.int64(1)}):
            with self.subTest(changes=list(changes)):
                self.write_arrays(**changes)
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_conversion_is_required_and_cannot_be_applied_twice(self):
        self.metadata['xchroma_conversion'] = None
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, '--xchroma-scale'):
            self.prepare()
        self.metadata['xchroma_conversion'] = dict(formula='log1p(signal * fixed_training_scale) / gp99',
                                                  fixed_training_scale=2, gp99=3)
        self.write_arrays(xchroma_input=np.log1p(self.normalized).astype(np.float32))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.prepare()

    def test_native_atac_path_keeps_original_normalization(self):
        atac_dir = self.directory / 'native'
        atac_dir.mkdir()
        native = np.arange(16, dtype=np.float16)
        native.tofile(atac_dir / 'chr1.atac.log1p.f16.memmap')
        (atac_dir / 'manifest.json').write_text(json.dumps({'chr1': {'length': 16}}))
        (atac_dir / '.global_p99').write_text('3\n')
        values, receipt = prepare_input(self.fasta, atac_dir, 'chr1', 0, 16)
        np.testing.assert_array_equal(values[4], native.astype(np.float32) / np.float32(3))
        self.assertEqual(receipt['global_p99'], 3)

    def test_cli_rejects_two_atac_inputs_and_double_normalization(self):
        common = ['--genome-fa', str(self.fasta), '--assembly', 'hg38', '--region', 'chr1:0-1000000',
                  '--weights-dir', 'unused', '--out', 'unused', '--atac-npz', str(self.path)]
        for extra in (['--atac-dir', 'unused'], ['--global-p99', '3']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(common + extra)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
