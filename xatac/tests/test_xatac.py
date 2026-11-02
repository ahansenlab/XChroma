"""Synthetic coordinate checks and a complete preprocessing/train/resume/predict test."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyBigWig
import pysam
import torch

from xatac.data import TrainingBatches, example, read_dataset
from xatac.io import decode_bf16, encode_bf16, sha256, write_json, write_npz
from xatac.loss import atac_loss, multinomial_loss, predictions_scaling, targets_scaling
from xatac.model import DNA2ATAC, load_weights
from xatac.predict import main as predict_main, predict
from xatac.sites import count_chrom, generate_sites, site_position
from xatac.train import configure, lr_for, train
from xatac.windows import build_windows, read_windows


def aligned(name, chrom, start, flag=99, mapq=60):
    read = pysam.AlignedSegment()
    read.query_name = name
    read.query_sequence = "A" * 50
    read.flag, read.reference_id, read.reference_start = flag, chrom, start
    read.mapping_quality, read.cigar = mapq, [(0, 50)]
    read.next_reference_id, read.next_reference_start, read.template_length = chrom, start + 100, 150
    read.query_qualities = pysam.qualitystring_to_array("I" * 50)
    return read


class XAtacTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bf16_and_loss(self):
        raw = np.array([0, 0.001, 1, 10, 100, 1e6], np.float32)
        expected = torch.from_numpy(raw).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(decode_bf16(encode_bf16(raw)), expected)
        values = torch.tensor([0., 1., 10., 100.])
        torch.testing.assert_close(predictions_scaling(targets_scaling(values, 2), 2), values)
        target = torch.tensor([1., 3., 0., 2.]).reshape(1, 4, 1)
        loss, total, positional = multinomial_loss(target, target, 4)
        self.assertAlmostEqual(float(total), 0)
        positive = np.array([1., 3., 2.])
        expected_pos = -np.sum(positive * np.log(positive / 6 + 1e-7)) / 4
        self.assertAlmostEqual(float(positional), expected_pos, places=5)
        repeated = multinomial_loss(target.repeat(3, 2, 1), target.repeat(3, 2, 1), 4)[0]
        torch.testing.assert_close(loss, repeated)
        self.assertGreater(float(multinomial_loss(target * 2, target, 4)[1]), 0)
        zero = torch.zeros_like(target)
        self.assertEqual(float(multinomial_loss(zero, zero, 4)[0]), 0)

    def test_site_coordinates_and_filters(self):
        bam = self.root / "test.bam"
        reads = [aligned("plus", 0, 10), aligned("minus", 0, 70, flag=147),
                 aligned("duplicate", 0, 120, flag=99 | 1024), aligned("low", 0, 150, mapq=10),
                 aligned("secondary", 0, 160, flag=99 | 256), aligned("supplementary", 0, 180, flag=99 | 2048),
                 aligned("improper", 0, 200, flag=65), aligned("qcfail", 0, 220, flag=99 | 512)]
        with pysam.AlignmentFile(bam, "wb", header={"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chrT", "LN": 1000}]}) as out:
            for read in reads:
                out.write(read)
        pysam.index(str(bam))
        with pysam.AlignmentFile(bam, "rb") as source:
            counts, qc = count_chrom([source], "chrT", 1000, 30, (0, 0))
        np.testing.assert_array_equal(np.flatnonzero(counts), [14, 115])
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(qc[0]["filtered"], 6)
        self.assertEqual(site_position(reads[0], 4, -4), 10)
        self.assertEqual(site_position(reads[1], 4, -4), 119)

    def test_short_targets_and_paired_augmentation(self):
        path = self.root / "record.npz"
        code = (np.arange(4096) % 5).astype(np.uint8)
        counts = np.arange(1024, dtype=np.float32)
        write_npz(path, sequence_code=code, atac_bf16=encode_bf16(counts))
        row = dict(path=path.name, length=4096, target_length=1024, target_offset=1536)
        x, y, observed = example(self.root, row, 2048, shift=128)
        xr, yr, reverse_observed = example(self.root, row, 2048, shift=128, reverse=True)
        self.assertEqual(observed, (384, 1408))
        self.assertEqual(reverse_observed, (640, 1664))
        np.testing.assert_array_equal(xr, x[::-1, ::-1])
        np.testing.assert_array_equal(yr, y[::-1])
        model = DNA2ATAC(seq_len=2048, width=16, n_dilated=1, up_channels=(16, 8, 8))
        prediction = model(torch.from_numpy(x[None]))
        loss, _ = atac_loss(prediction, torch.from_numpy(y[None]), model.track_mean, observed)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        state = model.state_dict()
        bad = {**state, "unrecognized.weight": torch.zeros(1)}
        with self.assertRaises(RuntimeError):
            load_weights(model, bad)

    def fixture(self):
        fasta, sizes = self.root / "genome.fa", self.root / "chrom.sizes"
        fasta.write_text("".join(f">chr{x}\n" + "ACGT" * 2048 + "\n" for x in ("T", "V", "X")))
        pysam.faidx(str(fasta))
        sizes.write_text("".join(f"chr{x}\t8192\n" for x in ("T", "V", "X")))
        bam = self.root / "library.bam"
        header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": f"chr{x}", "LN": 8192} for x in ("T", "V", "X")]}
        with pysam.AlignmentFile(bam, "wb", header=header) as out:
            for chrom in range(3):
                for j, start in enumerate(range(400, 1900, 100)):
                    out.write(aligned(f"{chrom}_{j}", chrom, start, 99 if j % 2 else 147))
        pysam.index(str(bam))
        bw = self.root / "sites.bw"
        meta = generate_sites([bam], sizes, bw, "synthetic", (0, 0), normalization_total=1000)
        self.assertEqual(meta["raw_site_count"], 45)
        self.assertAlmostEqual(meta["sum_float32"], 1000, places=3)
        with pyBigWig.open(str(bw)) as source:
            signal = np.nan_to_num(source.values("chrT", 0, 8192, numpy=True))
            self.assertEqual(np.count_nonzero(signal), 15)
        bed = self.root / "windows.bed"
        bed.write_text("chrT\t512\t1536\tTRAIN\nchrV\t512\t1536\tVALID\nchrX\t512\t1536\tTEST\n")
        data = self.root / "data"
        build_windows(fasta, bw, bed, data, seq_len=1024, flank=256)
        config = self.root / "config.json"
        write_json(config, dict(model=dict(seq_len=1024, width=16, n_dilated=1, up_channels=[16, 8, 8]),
                                batch_size=2, max_steps=4, warmup=1, loader_workers=0, threads=1,
                                shift_bp=32, val_windows=1, validate_every=2, save_every=1, precision="fp32"))
        return fasta, data, config

    def test_complete_pipeline_and_exact_resume(self):
        fasta, data, config_path = self.fixture()
        meta, rows, _ = read_dataset(data)
        config = configure(json.loads(config_path.read_text()), meta)
        training = [r for r in rows if r["subset"] == "TRAIN"]
        a = TrainingBatches(data, training, config, 1, 4)
        b = TrainingBatches(data, training, config, 3, 4)
        torch.testing.assert_close(a[2]["x"], b[0]["x"], rtol=0, atol=0)
        torch.testing.assert_close(a[2]["raw"], b[0]["raw"], rtol=0, atol=0)
        continuous, resumed = self.root / "continuous", self.root / "resumed"
        train(data, continuous, config_path, "cpu")
        train(data, resumed, config_path, "cpu", stop_after=2)
        train(data, resumed, config_path, "cpu", resume=True)
        first = torch.load(continuous / "models/latest.pt", weights_only=False)
        second = torch.load(resumed / "models/latest.pt", weights_only=False)
        self.assertEqual(first["step"], second["step"])
        self.assertEqual(first["samples"], second["samples"])
        for key in first["model"]:
            torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
        for key in first["optimizer"]["state"]:
            for name in first["optimizer"]["state"][key]:
                torch.testing.assert_close(first["optimizer"]["state"][key][name],
                                           second["optimizer"]["state"][key][name], rtol=0, atol=0)
        self.assertEqual(first["optimizer"]["param_groups"][0]["lr"], lr_for(4, config))
        result = predict([resumed / "models/step4.pt"], fasta, "chrV", 524, 1524,
                         self.root / "prediction.npz", "cpu", "synthetic", 2, 5)
        self.assertEqual(result.shape, (1000,))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all())
        with np.load(self.root / "prediction.npz") as z:
            np.testing.assert_array_equal(z["xchroma_input"], (np.log1p(result.astype(np.float64) * 2) / 5).astype(np.float32))
        changed = json.loads(config_path.read_text()); changed["lr"] = 1e-3
        write_json(config_path, changed)
        with self.assertRaises(ValueError):
            train(data, resumed, config_path, "cpu", resume=True)
        ds = json.loads((data / "dataset.json").read_text()); ds["signal"] = "fragment_coverage"
        write_json(data / "dataset.json", ds)
        with self.assertRaises(ValueError):
            read_dataset(data)

    def test_prediction_scale_keyword_aliases_export_only_canonical_names(self):
        fasta = self.root / "genome.fa"
        fasta.write_text(">chr1\n" + "ACGT" * 512 + "\n")
        pysam.faidx(str(fasta))
        config = dict(seq_len=1024, width=16, n_dilated=1, up_channels=[16, 8, 8])
        model = DNA2ATAC(**config)
        checkpoint = self.root / "model.pt"
        torch.save(dict(config=dict(model=config), model=model.state_dict(), step=0), checkpoint)
        reference = None
        for index, scales in enumerate((dict(xchroma_scale=2), dict(xchrom_scale=2),
                                        dict(xchroma_scale=2, xchrom_scale=2))):
            with self.subTest(scales=scales):
                output = self.root / f"prediction_{index}.npz"
                signal = predict([checkpoint], fasta, "chr1", 524, 1524, output,
                                 "cpu", "synthetic", gp99=5, **scales)
                if reference is None:
                    reference = signal
                np.testing.assert_array_equal(signal, reference)
                with np.load(output) as arrays:
                    self.assertNotIn("xchrom_input", arrays)
                    np.testing.assert_array_equal(arrays["xchroma_input"],
                        (np.log1p(signal.astype(np.float64) * 2) / 5).astype(np.float32))
                metadata = json.loads(Path(str(output) + ".json").read_text())
                self.assertNotIn("xchrom_conversion", metadata)
                self.assertEqual(metadata["xchroma_conversion"]["fixed_training_scale"], 2)
                self.assertEqual(metadata["output_sha256"], sha256(output))
        with self.assertRaisesRegex(ValueError, "Conflicting values for --xchroma-scale"):
            predict([checkpoint], fasta, "chr1", 524, 1524, self.root / "conflict.npz",
                    "cpu", "synthetic", xchroma_scale=2, xchrom_scale=3, gp99=5)
        self.assertFalse((self.root / "conflict.npz").exists())

    def test_prediction_cli_accepts_aliases_and_rejects_conflicts(self):
        common = ["--checkpoint", "model.pt", "--fasta", "genome.fa", "--assembly", "synthetic",
                  "--chrom", "chr1", "--start", "524", "--end", "1524", "--output", "out.npz",
                  "--device", "cpu", "--gp99", "5"]
        for flags in (["--xchroma-scale", "2"], ["--xchrom-scale", "2"],
                      ["--xchroma-scale", "2", "--xchrom-scale", "2"]):
            with self.subTest(flags=flags), patch("xatac.predict.predict") as run:
                predict_main(common + flags)
                self.assertEqual(run.call_args.args[8:10], (2, 5))
        with patch("xatac.predict.predict") as run, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            predict_main(common + ["--xchroma-scale", "2", "--xchrom-scale", "3"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()
        help_text = io.StringIO()
        with contextlib.redirect_stdout(help_text), self.assertRaises(SystemExit) as error:
            predict_main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--xchroma-scale", help_text.getvalue())
        self.assertNotIn("--xchrom-scale", help_text.getvalue())

    def test_split_overlap_rejected(self):
        bed = self.root / "bad.bed"
        bed.write_text("chr1\t1000\t2024\tTRAIN\nchr1\t2024\t3048\tVALID\n")
        with self.assertRaises(ValueError):
            read_windows(bed, 1024, 256)

    def test_spawn_workers_preserve_data_order(self):
        _, data, config_path = self.fixture()
        meta, rows, _ = read_dataset(data)
        config = configure(json.loads(config_path.read_text()), meta)
        training = [row for row in rows if row["subset"] == "TRAIN"]
        batches = TrainingBatches(data, training, config, 1, 4)
        expected = [batches[i] for i in range(len(batches))]
        loader = torch.utils.data.DataLoader(
            batches, batch_size=None, num_workers=2, prefetch_factor=1,
            multiprocessing_context="spawn", generator=torch.Generator().manual_seed(42))
        actual = list(loader)
        self.assertEqual(len(actual), len(expected))
        for serial, parallel in zip(expected, actual):
            self.assertEqual(serial.keys(), parallel.keys())
            for key in serial:
                if isinstance(serial[key], torch.Tensor):
                    torch.testing.assert_close(serial[key], parallel[key], rtol=0, atol=0)
                else:
                    self.assertEqual(serial[key], parallel[key])


if __name__ == "__main__":
    unittest.main()
