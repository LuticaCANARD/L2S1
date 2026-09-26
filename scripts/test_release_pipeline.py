from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from prepare_cargo_release import prepare
from prepare_release import ROOT, versions, validate_npm, seal
from publish_packages import digest, publish_npm, publish_cargo, check_pypi, registry


class ReleasePipeline(unittest.TestCase):
    def test_versions_and_lock_must_match_stable_tag(self):
        import tomllib
        version = tomllib.loads((ROOT / 'Cargo.toml').read_text())['package']['version']
        self.assertEqual(versions(ROOT, 'v' + version), version)
        major, minor, patch_version = map(int, version.split('.'))
        different = f'v{major}.{minor}.{patch_version + 1}'
        for tag in [different, 'v' + version + '-rc.1', 'main', 'v' + version + ';false']:
            with self.assertRaises(ValueError):
                versions(ROOT, tag)

    def test_cargo_staging_preserves_checkout_and_excludes_only_unpublished_wgpu(self):
        before = (ROOT / 'Cargo.toml').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'source'
            prepare(ROOT, output)
            import tomllib
            manifest = tomllib.loads((output / 'Cargo.toml').read_text())
            self.assertNotIn('rullama-engine', manifest['dependencies'])
            self.assertNotIn('wgpu', manifest['features'])
            for feature in ['llama', 'llama-cuda', 'llama-metal', 'openrouter']:
                self.assertIn(feature, manifest['features'])
            self.assertTrue((output / 'src/stdio.rs').is_file())
            self.assertNotIn('feature = "wgpu"', (output / 'src/lib.rs').read_text())
            with self.assertRaises(ValueError):
                prepare(ROOT, output)
        self.assertEqual((ROOT / 'Cargo.toml').read_bytes(), before)

    def test_runtime_archive_manifest_is_checked_against_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.tgz'
            contents = {'package/package.json': json.dumps({'name':'@l2s1/runtime-linux-x64','version':'0.1.0'}).encode(),
                'package/runtime-manifest.json': json.dumps({'platform':'linux-x64','version':'0.1.0','files':[
                    {'path':'bin/l2s1','bytes':3,'sha256':'bad'}]}).encode(), 'package/bin/l2s1': b'abc'}
            with tarfile.open(path, 'w:gz') as archive:
                for name, value in contents.items():
                    info = tarfile.TarInfo(name); info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate_npm(path, '@l2s1/runtime-linux-x64', '0.1.0', 'linux-x64')

    def test_registry_auth_or_server_error_is_never_assumed_missing(self):
        for status in [401, 403, 429, 500]:
            with patch('publish_packages.urlopen', side_effect=HTTPError('url', status, 'failure', {}, None)):
                with self.assertRaises(HTTPError):
                    registry('https://example.test')
        with patch('publish_packages.urlopen', side_effect=HTTPError('url', 404, 'missing', {}, None)):
            self.assertIsNone(registry('https://example.test'))

    def test_npm_order_and_checksum_guard_make_retries_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['l2s1-runtime-linux-x64-0.1.0.tgz','l2s1-node-0.1.0.tgz']:
                (root / name).write_bytes(name.encode())
            with patch('publish_packages.registry', return_value=None) as lookup, patch('publish_packages.subprocess.run') as run:
                publish_npm(root, '0.1.0')
                calls = [item.args[0][2] for item in run.call_args_list]
                self.assertIn('runtime-linux-x64', calls[0])
                self.assertIn('node-0.1.0', calls[1])
                self.assertEqual(lookup.call_args_list[0].args[0], 'https://registry.npmjs.org/%40l2s1%2Fruntime-linux-x64/0.1.0')
                self.assertEqual(lookup.call_args_list[1].args[0], 'https://registry.npmjs.org/%40l2s1%2Fnode/0.1.0')
            path = root / 'l2s1-runtime-linux-x64-0.1.0.tgz'
            matching = {'dist': {'integrity':'sha512-'+base64.b64encode(bytes.fromhex(digest(path,'sha512'))).decode()}}
            with patch('publish_packages.registry', side_effect=[matching,None]), patch('publish_packages.subprocess.run') as run:
                publish_npm(root, '0.1.0')
                self.assertEqual(run.call_count, 1)
            with patch('publish_packages.registry', return_value={'dist':{'integrity':'bad'}}), patch('publish_packages.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    publish_npm(root, '0.1.0')
                run.assert_not_called()

    def test_existing_pypi_files_must_match_sealed_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'l2s1_sdk-0.1.0.tar.gz').write_bytes(b'source')
            matching = {'urls':[{'filename':'l2s1_sdk-0.1.0.tar.gz','digests':{'sha256':digest(root/'l2s1_sdk-0.1.0.tar.gz')}}]}
            with patch('publish_packages.registry', return_value=matching) as lookup:
                check_pypi(root,'0.1.0')
                self.assertEqual(lookup.call_args.args[0], 'https://pypi.org/pypi/l2s1-sdk/0.1.0/json')
            matching['urls'][0]['digests']['sha256'] = 'bad'
            with patch('publish_packages.registry', return_value=matching):
                with self.assertRaises(ValueError):
                    check_pypi(root,'0.1.0')
            seal(root,[root/'l2s1_sdk-0.1.0.tar.gz'],'v0.1.0','a'*40)
            self.assertIn('a'*40,(root/'release.json').read_text())

    def test_cargo_publication_order_and_repack_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'assets'; assets.mkdir()
            source = root / 'source'; packages = source / 'target/package'; packages.mkdir(parents=True)
            for crate in ['l2s1-llama-sys', 'l2s1']:
                name = f'{crate}-0.1.0.crate'
                (assets/name).write_bytes(name.encode())
                (packages/name).write_bytes(name.encode())
            with patch('publish_packages.registry', return_value=None), patch('publish_packages.subprocess.run') as run:
                publish_cargo(assets, '0.1.0', source)
                commands = [item.args[0] for item in run.call_args_list]
                self.assertEqual([command[4] for command in commands], ['l2s1-llama-sys']*2+['l2s1']*2)
                self.assertEqual([command[1] for command in commands], ['package', 'publish']*2)
            (packages/'l2s1-llama-sys-0.1.0.crate').write_bytes(b'different')
            with patch('publish_packages.registry', return_value=None), patch('publish_packages.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    publish_cargo(assets, '0.1.0', source)
                self.assertEqual(run.call_count, 1)  # Only package verification, no upload.


if __name__ == '__main__':
    unittest.main()
