"""Describe the locked main ensemble or verify all ten local weight files; no ML imports."""
import argparse
import hashlib
import json
from pathlib import Path
import re

MANIFEST = Path(__file__).with_name('current_model.json')
STEPS = list(range(30500, 35001, 500))


def read_manifest(path=MANIFEST):
    data = json.loads(Path(path).read_text())
    members = data['members']
    if data['release_id'] != 'xchrom_avg10_present_30500_35000' or [m['step'] for m in members] != STEPS:
        raise ValueError('Manifest must specify the locked ten steps 30500–35000 every 500.')
    if len({m['filename'] for m in members}) != 10:
        raise ValueError('Every ensemble member requires a distinct filename.')
    for m in members:
        if m['weight'] != 0.1 or m['bytes'] <= 0 or not re.fullmatch(r'[a-f0-9]{64}', m['sha256']):
            raise ValueError('Invalid member weight, byte count or SHA256.')
        if Path(m['filename']).name != m['filename']:
            raise ValueError('Member filename must not include a directory.')
    return data


def verify_weights(directory, data):
    directory = Path(directory).expanduser().resolve()
    missing = [m['filename'] for m in data['members'] if not (directory / m['filename']).is_file()]
    if missing:
        raise ValueError('Missing locked ensemble members: ' + ', '.join(missing))
    verified = []
    for m in data['members']:
        path = directory / m['filename']
        if path.stat().st_size != m['bytes']:
            raise ValueError(f'Byte-count mismatch: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != m['sha256']:
            raise ValueError(f'SHA256 mismatch: {path}')
        verified.append({'step': m['step'], 'path': str(path), 'sha256': m['sha256']})
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--describe', action='store_true', help='Show manifest identity and availability; does not verify local weights.')
    action.add_argument('--weights-dir', type=Path, help='Directory containing every exact named weight file; verify actual bytes.')
    args = parser.parse_args()
    try:
        data = read_manifest()
        if args.describe:
            print(json.dumps({'release_id': data['release_id'], 'steps': STEPS, 'total_bytes': sum(m['bytes'] for m in data['members']), 'local_weights_verified': False, 'availability': data['availability']}, indent=2))
        else:
            verified = verify_weights(args.weights_dir, data)
            print(json.dumps({'release_id': data['release_id'], 'local_weights_verified': True, 'member_count': len(verified), 'members': verified, 'inference_executed': False}, indent=2))
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
