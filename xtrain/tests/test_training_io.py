"""Frozen numerical regression cases for training readers and data.make_sampler."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cooler
import h5py
import numpy as np
import pandas as pd
import pyfaidx

from xtrain import data
from xtrain.x_utils import Genomic2DFeatures, MemmapGenome


FIXTURES = Path(__file__).parent / 'fixtures'
CHROMS = [('chr2', 6000), ('chr10', 5200), ('chr1', 4800), ('chrX', 5600)]
QUERIES = [('square10', ('chr1', 0, 1000), None),
           ('small3x6', ('chr1', 0, 300), ('chr2', 0, 600)),
           ('tall12x5', ('chr1', 0, 1200), ('chrX', 100, 600)),
           ('wide6x13', ('chrX', 0, 600), ('chr2', 100, 1400))]


class TrainingIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.fasta, cls.cool = cls.root / 'genome.fa', cls.root / 'contacts.cool'
        rng = np.random.RandomState(81026)
        with cls.fasta.open('w') as out:
            for chrom, length in CHROMS:
                sequence = ''.join(rng.choice(list('ACGT'), length))
                sequence = 'ACGTNRYKMSWBDHVnacgt.-' + sequence[22:]
                out.write(f'>{chrom}\n{sequence}\n')
        pyfaidx.Fasta(str(cls.fasta)).close()
        cls.blacklist = cls.root / 'blacklist.bed'
        cls.blacklist.write_text('chr1\t800\t1000\nchrX\t2000\t2100\n')
        bins = pd.DataFrame([(c, s, s + 100) for c, n in CHROMS for s in range(0, n, 100)],
                            columns=['chrom', 'start', 'end'])
        i, j = np.triu_indices(len(bins))
        counts = ((i * 13 + j * 7 + (i-j)**2) % 19).astype(np.int32)
        counts[(i+j) % 11 == 0] = 0
        cooler.create_cooler(str(cls.cool), bins,
                             pd.DataFrame(dict(bin1_id=i, bin2_id=j, count=counts)), ordered=True)
        weights = 0.5 + (np.arange(len(bins)) % 13) / 16
        weights[np.arange(len(bins)) % 17 == 0] = np.nan
        with h5py.File(cls.cool, 'r+') as handle:
            handle['bins'].create_dataset('weight', data=weights)
        cls.atac_dir = cls.root / 'atac'
        cls.atac_dir.mkdir()
        for index, (chrom, length) in enumerate(CHROMS):
            values = ((np.arange(length) * 7 % 53) / 16 + index / 8).astype(np.float16)
            values.tofile(cls.atac_dir / f'{chrom}.atac.log1p.f16.memmap')
        (cls.atac_dir / 'manifest.json').write_text(json.dumps({'chroms': dict(CHROMS)}))
        (cls.atac_dir / '.global_p99').write_text('4.000000\n')
        cls.expected = np.load(FIXTURES / 'training_io_reference.npz')
        cls.coordinates = json.loads((FIXTURES / 'training_io_coordinates.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.expected.close()
        cls.tmp.cleanup()

    def setUp(self):
        config = patch.multiple(data, ATAC_NORM='global', ATAC_NORM_SCALE=1.0,
                                ATAC_ZERO=False, _GP99=None)
        config.start()
        self.addCleanup(config.stop)
        self.atac = data._load_atac_memmaps(str(self.atac_dir))
        data.bind_gp99(str(self.atac_dir), self.atac)

    def genome(self, cache=None):
        return MemmapGenome(str(self.fasta), blacklist_regions=str(self.blacklist), memmapfile=cache)

    def target(self, cg=True):
        return Genomic2DFeatures(str(self.cool), 'synthetic', (10, 10), cg=cg)

    def sampler(self, holdouts=True):
        return data.make_sampler(self.genome(), self.target(), 'synthetic', 1000, 100,
                                 ['chr2'] if holdouts else [], ['chr10'] if holdouts else [], self.atac)

    def assertFrozen(self, name, actual):
        expected = self.expected[name]
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(actual, expected)

    def test_dna_ambiguity_strand_padding_and_cache_bytes(self):
        genome = self.genome()
        self.assertEqual(genome.get_chr_lens(), sorted(CHROMS))
        for name, start, end, strand, pad in [('bases', 0, 22, '+', False),
                ('reverse', 0, 22, '-', False), ('pad_left', -7, 9, '+', True),
                ('pad_right', 4791, 4808, '+', True), ('pad_inbounds', 30, 45, '+', True)]:
            self.assertFrozen('dna_' + name, genome.get_encoding_from_coords('chr1', start, end, strand, pad))
        cache = self.root / 'genome.mmap'
        self.genome(str(cache)).get_encoding_from_coords('chrX', 40, 90)
        self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(),
                         'b6265f07c6724bc3195b81878e612750bb811b0c59d46e0aacd3df692fa69d85')
        self.assertFrozen('cache_loaded_dna', self.genome(str(cache)).get_encoding_from_coords('chrX', 40, 90))

    def test_blacklist_boundary_rules(self):
        g = self.genome()
        for start, end, valid in [(700, 800, True), (799, 801, False),
                                  (850, 900, False), (1000, 1100, True), (-1, 10, False)]:
            self.assertEqual(g.coords_in_bounds('chr1', start, end), valid)
        self.assertFalse(g.coords_in_bounds('absent', 0, 10))

    def test_balanced_coarsegrain_rectangles_and_missing_values(self):
        for cg in [False, True]:
            for name, first, second in QUERIES:
                kwargs = {} if second is None else dict(zip(['chrom2', 'start2', 'end2'], second))
                self.assertFrozen(f'contact_cg{int(cg)}_{name}', self.target(cg).get_feature_data(*first, **kwargs))

    def test_coordinate_stream_and_chromosome_holdouts(self):
        sampler = self.sampler()
        for mode in ['train', 'validate', 'test']:
            for batch in range(2):
                self.assertEqual(sampler.sample(12, mode=mode, coordinate_only=True),
                                 self.coordinates[f'{mode}_{batch}'])
        sampler = self.sampler(holdouts=False)
        self.assertEqual(sampler.modes, ['train'])

    def test_make_sampler_dna_only_jitter_atac_channel_and_targets(self):
        sampler = self.sampler()
        for batch in range(2):
            sequences, targets = sampler.sample(4)
            self.assertFrozen(f'batch{batch}_sequences', sequences)
            self.assertFrozen(f'batch{batch}_targets', targets)
        # The first DNA query starts at -1; its ATAC starts at genomic zero.
        first = self.expected['batch0_sequences'][0]
        np.testing.assert_array_equal(first[0, :4], np.full(4, 0.25))
        self.assertEqual(first[0, 4], float(self.atac['chrX'][0]) / 4.0)


if __name__ == '__main__':
    unittest.main()
