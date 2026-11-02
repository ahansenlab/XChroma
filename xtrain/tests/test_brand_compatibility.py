"""Check XChroma names and compatibility with existing imports and weights."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from xtrain import model
from xtrain.checkpoints.verify import read_manifest, verify_weights


class BrandCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = read_manifest()
        self.payloads = {}
        for member in self.manifest['members']:
            payload = f"synthetic checkpoint {member['step']}".encode()
            member.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            self.payloads[member['filename']] = payload

    def write_weights(self, directory, legacy=False):
        directory.mkdir(exist_ok=True)
        paths = []
        for name, payload in self.payloads.items():
            if legacy:
                name = name.replace('xchroma_', 'xchrom_', 1)
            path = directory / name
            path.write_bytes(payload)
            paths.append(path)
        return paths

    def test_current_and_legacy_filenames(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                directory = self.root / str(legacy)
                paths = self.write_weights(directory, legacy=legacy)
                verified = verify_weights(directory, self.manifest)
                self.assertEqual([Path(m['path']) for m in verified], [path.resolve() for path in paths])
                self.assertEqual([m['step'] for m in verified],
                                 [m['step'] for m in self.manifest['members']])

    def test_current_name_takes_precedence_and_corruption_is_rejected(self):
        paths = self.write_weights(self.root)
        self.write_weights(self.root, legacy=True)
        self.assertEqual(Path(verify_weights(self.root, self.manifest)[0]['path']), paths[0].resolve())
        paths[0].write_bytes(b'!' * paths[0].stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify_weights(self.root, self.manifest)

    def test_legacy_files_require_exact_size_and_hash(self):
        path = self.write_weights(self.root, legacy=True)[0]
        payload = path.read_bytes()
        for changed, error in ((payload + b'!', 'Byte-count mismatch'),
                               (b'!' * len(payload), 'SHA256 mismatch')):
            with self.subTest(error=error):
                path.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, error):
                    verify_weights(self.root, self.manifest)

    def test_missing_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing locked ensemble members'):
            verify_weights(self.root, self.manifest)

    def test_current_and_legacy_manifest_ids(self):
        path = self.root / 'manifest.json'
        for prefix in ('xchroma', 'xchrom'):
            self.manifest['release_id'] = prefix + '_avg10_present_30500_35000'
            path.write_text(json.dumps(self.manifest))
            self.assertEqual(read_manifest(path)['release_id'], self.manifest['release_id'])
        self.manifest['release_id'] = 'unrecognized_model'
        path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'locked ten steps'):
            read_manifest(path)

    def test_class_alias_and_state_dict_compatibility(self):
        self.assertIs(model.XChromNet, model.XChromaNet)
        with contextlib.redirect_stdout(io.StringIO()), torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            legacy = model._replace_bn_with_gn(model.XChromNet())
            torch.manual_seed(42)
            current = model.build_net()
        self.assertIsInstance(current, model.XChromaNet)
        old_state, new_state = legacy.state_dict(), current.state_dict()
        self.assertEqual(old_state.keys(), new_state.keys())
        for name, value in old_state.items():
            self.assertTrue(torch.equal(value, new_state[name]), name)
        loaded = current.load_state_dict(old_state, strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])

    def test_architecture_hash_matches(self):
        digest = hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
        self.assertEqual(read_manifest()['architecture']['sha256'], digest)


if __name__ == '__main__':
    unittest.main()
