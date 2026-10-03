"""Generate C.Origami benchmark caches from explicit prepared inputs."""
from pathlib import Path
import argparse
import hashlib
import json


def selfnorm_log(matrix):
    import numpy as np
    nb = matrix.shape[0]
    out = np.full_like(matrix, np.nan, dtype=np.float64)
    for distance in range(nb):
        index = np.arange(nb - distance)
        values = matrix[index, index + distance]
        mean = np.nanmean(values[np.isfinite(values) & (values > 0)]) if np.isfinite(values).any() else np.nan
        if np.isfinite(mean) and mean > 0:
            out[index, index + distance] = values / mean
            out[index + distance, index] = out[index, index + distance]
    return np.log(np.where(out > 0, out, np.nan))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dna', type=Path, required=True)
    parser.add_argument('--ctcf', type=Path, required=True)
    parser.add_argument('--atac', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True, help='Reference CELL_chrN.maps.npz files; only starts are read.')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chrom')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if not args.dna.is_dir() or not args.windows.is_dir():
        parser.error('--dna and --windows must be existing directories')
    for path in (args.checkpoint, args.ctcf, args.atac):
        if not path.is_file(): parser.error(f'Missing input: {path}')
    if args.limit < 0: parser.error('--limit must be nonnegative')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('--out must be empty')
    files = sorted(args.windows.glob(f'{args.cell}_chr*.maps.npz'))
    if args.chrom: files = [p for p in files if p.name.endswith(f'_{args.chrom}.maps.npz')]
    if not files: parser.error('No matching window manifests found')
    import numpy as np
    import torch
    from skimage.transform import resize
    import corigami.inference.utils.inference_utils as iu
    model = iu.load_default(str(args.checkpoint))
    win_input, win_output, nb = 2_097_152, 2_000_000, 200
    crop = int(round((win_input - win_output) / 2 / (win_input / 256)))
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for file in files:
        chrom = file.name.split('_')[-1].split('.')[0]
        with np.load(file, allow_pickle=False) as source:
            starts = source['starts'].astype(int)
        if args.limit: starts = starts[:args.limit]
        seq, ctcf, atac = iu.load_data_default(chrom, str(args.dna), str(args.ctcf), str(args.atac))
        maps = []
        for start in starts:
            center = int(start) + win_output // 2
            sr, cr, ar = iu.get_data_at_interval(chrom, center - win_input // 2, center + win_input // 2, seq, ctcf, atac)
            if np.shape(sr)[0] != win_input: raise ValueError(f'Incomplete sequence at {chrom}:{start}')
            with torch.no_grad():
                prediction = model(iu.preprocess_default(sr, cr, ar))[0].detach().cpu().numpy().astype(np.float64)
            prediction = np.expm1(np.clip(prediction, 0, None))
            maps.append(selfnorm_log(resize(prediction[crop:256-crop, crop:256-crop], (nb, nb), anti_aliasing=True, preserve_range=True)))
        out = args.out / f'{args.cell}_{chrom}.maps.npz'
        np.savez_compressed(out, corigami=np.array(maps, dtype=np.float16), starts=starts, res=10_000)
        records.append({'chrom': chrom, 'windows': len(starts), 'source_windows': str(file.resolve()), 'output': out.name})
        print(f'{chrom}: {len(starts)} windows -> {out}', flush=True)
        del seq, ctcf, atac
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
        return value.hexdigest()
    (args.out / 'prediction_manifest.json').write_text(json.dumps({
        'cell': args.cell, 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': digest(args.checkpoint), 'dna': str(args.dna.resolve()),
        'ctcf': str(args.ctcf.resolve()), 'atac': str(args.atac.resolve()),
        'records': records, 'output_space': 'natural_log_observed_expected',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
