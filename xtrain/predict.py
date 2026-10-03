#!/usr/bin/env python3
"""Predict a 1-Mb contact map at 1-kb resolution using the current XChrom ensemble."""
import argparse
import json
from pathlib import Path
import platform
import re
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xtrain.checkpoints.verify import read_manifest, verify_weights


def parse_region(value):
    match = re.fullmatch(r'([^:]+):(\d+)-(\d+)', value.replace(',', ''))
    if not match:
        raise argparse.ArgumentTypeError('Use chrom:start-end in 0-based, half-open coordinates')
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if end - start != 1_000_000 or start < 0 or start % 1000:
        raise argparse.ArgumentTypeError('Require a 1,000,000-bp interval starting on a 1-kb boundary')
    return chrom, start, end


def clean_state_dict(payload):
    state = payload.get('model', payload.get('state_dict', payload))
    clean = {}
    for key, value in state.items():
        while key.startswith(('module.', '_orig_mod.')):
            key = key.split('.', 1)[1]
        if key in clean:
            raise ValueError(f'Duplicate checkpoint key after prefix removal: {key}')
        clean[key] = value
    return clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--genome-fa', type=Path, required=True, help='Uncompressed FASTA with matching .fai index')
    parser.add_argument('--atac-dir', type=Path, required=True, help='Native log1p ATAC AZ/memmap directory and manifest.json')
    parser.add_argument('--global-p99', type=float, help='Source-global ATAC normalization; otherwise read .global_p99')
    parser.add_argument('--assembly', required=True, help='Genome assembly recorded in the output, e.g. hg38 or mm39')
    parser.add_argument('--region', type=parse_region, required=True, help='0-based half-open chrom:start-end')
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are not overwritten')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=8, help='CPU intra-op threads')
    parser.add_argument('--check-only', action='store_true', help='Verify all weights and prepare input without model inference')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    try:
        manifest = read_manifest()
        verified = verify_weights(args.weights_dir, manifest)
        from xtrain.inputs import prepare_input, sha256
        import numpy as np
        model_sha = sha256(Path(__file__).with_name('model.py'))
        if model_sha != manifest['architecture']['sha256']:
            raise ValueError('model.py differs from the architecture bound to this checkpoint manifest')
        chrom, start, end = args.region
        x, input_receipt = prepare_input(args.genome_fa, args.atac_dir, chrom, start, end, args.global_p99)
        if x.shape != (5, 1_000_000) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Prepared input must be finite float32 with shape (5, 1000000)')
        args.out.mkdir(parents=True, exist_ok=False)
        record = {
            'status': 'INPUTS_VERIFIED', 'model_id': manifest['release_id'],
            'assembly': args.assembly, 'chrom': chrom, 'start': start, 'end': end,
            'coordinates': '0-based half-open', 'resolution_bp': 1000,
            'input': input_receipt, 'members': verified, 'inference_executed': False,
            'model_source_sha256': model_sha,
            'input_source_sha256': sha256(Path(__file__).with_name('inputs.py')),
            'runner_sha256': sha256(__file__), 'completed_members': [],
        }
        receipt_path = args.out / 'manifest.json'

        def save():
            temporary = receipt_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
            temporary.replace(receipt_path)

        save()
        if args.check_only:
            print(f'Inputs and all ten checkpoint files verified: {receipt_path}')
            return
        import torch
        from xtrain.model import build_net
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        net = build_net().float().to(args.device).eval()
        tensor = torch.from_numpy(x[None].copy()).to(args.device)
        total = np.zeros((1000, 1000), dtype=np.float64)
        record['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                             'torch': torch.__version__, 'device': args.device, 'threads': args.threads,
                             'forward_dtype': 'float32', 'accumulation_dtype': 'float64',
                             'autocast': False, 'tf32': False}
        record['status'] = 'RUNNING'
        save()
        try:
            for member in verified:
                payload = torch.load(member['path'], map_location='cpu', weights_only=True)
                net.load_state_dict(clean_state_dict(payload), strict=True)
                del payload
                started = time.monotonic()
                with torch.no_grad(), torch.autocast(device_type=args.device, enabled=False):
                    prediction = net(tensor)
                if prediction.shape != (1, 1, 1000, 1000) or prediction.dtype != torch.float32:
                    raise ValueError('Unexpected model output shape or dtype')
                values = prediction.detach().cpu().numpy()[0, 0]
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite prediction at step {member["step"]}')
                total += values.astype(np.float64)
                record['completed_members'].append({'step': member['step'], 'strict_load': True,
                    'elapsed_seconds': time.monotonic() - started})
                save()
                print(f'Completed ensemble member {len(record["completed_members"])}/10', flush=True)
                del prediction, values
            mean = total / 10
            output = args.out / 'prediction.npz'
            np.savez_compressed(output, prediction=mean, chrom=np.array(chrom), start=np.array(start),
                                end=np.array(end), bin_bp=np.array(1000), assembly=np.array(args.assembly),
                                value_space=np.array('native ln(O/E)'),
                                model_id=np.array(manifest['release_id']))
            record.update(status='COMPLETE', inference_executed=True,
                          output={'file': output.name, 'sha256': sha256(output), 'shape': [1000, 1000],
                                  'value_space': 'native ln(O/E)', 'dtype': 'float64',
                                  'ensemble': 'arithmetic mean of ten output maps',
                                  'symmetry_max_abs': float(np.max(np.abs(mean - mean.T)))})
            save()
            print(f'Prediction: {output}')
        except BaseException as exc:
            record.update(status='FAILED', error=str(exc))
            save()
            raise
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
