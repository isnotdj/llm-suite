"""Offline packaging selects the intended image and rejects wrong architectures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OfflinePackageTests(unittest.TestCase):
    def test_dockerfiles_keep_build_targets_independent(self):
        suite = (ROOT/'Dockerfile').read_text()
        native = (ROOT/'Dockerfile.native').read_text()
        self.assertIn('FROM ubuntu:24.04',suite)
        self.assertNotIn('vllm/vllm-openai',suite)
        self.assertNotIn('ARG PIP_INDEX_URL',native)
        self.assertNotIn('ARG PIP_INDEX_URL',suite)
        self.assertNotIn('ARG TORCH_INDEX_URL',suite)
        self.assertIn('--mount=type=secret,id=pip_index_url',native)
        self.assertIn('--mount=type=secret,id=torch_index_url',suite)

    def test_image_selection_platform_and_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root/'docker'
            tool.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$DOCKER_CALLS"
case "$1" in
  build) printf '%s\n' "$DOCKER_BUILDKIT" >> "$BUILDERS"; exit 0 ;;
  image) printf '%s\\n' "$MOCK_PLATFORM" ;;
  save) printf 'fixture-image-%s' "$2" ;;
  load) cat > "$LOADED_IMAGE" ;;
  *) exit 1 ;;
esac
''')
            tool.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       DOCKER_CALLS=str(root/'calls'), MOCK_PLATFORM='linux/amd64',
                       LOADED_IMAGE=str(root/'loaded'), BUILDERS=str(root/'builders'), PIP_INDEX_URL='https://mirror.invalid/simple',
                       TORCH_INDEX_URL='https://mirror.invalid/torch')
            package = ROOT/'scripts/package_offline.sh'
            for mode in ('suite','native','all'):
                out = root/mode
                subprocess.run([str(package),str(out),mode],env=env,check=True,capture_output=True)
                calls = (root/'calls').read_text().splitlines()
                builds = [line for line in calls if line.startswith('build ')]
                self.assertTrue(all('--secret id=pip_index_url,env=PIP_INDEX_URL' in line for line in builds))
                self.assertTrue(all('--secret id=torch_index_url,env=TORCH_INDEX_URL' in line for line in builds))
                self.assertTrue(all('mirror.invalid' not in line for line in builds))
                self.assertTrue(all(value == '1' for value in (root/'builders').read_text().splitlines()))
                if mode == 'suite':
                    self.assertEqual(len(builds),1)
                    self.assertNotIn('Dockerfile.native',builds[0])
                if mode == 'native':
                    self.assertEqual(len(builds),1)
                    self.assertIn('Dockerfile.native',builds[0])
                subprocess.run([str(out/'load_offline.sh')],env=env,check=True,capture_output=True)
                self.assertTrue((root/'loaded').read_bytes().startswith(b'fixture-image-'))
                (root/'calls').unlink()
                (root/'builders').unlink()

            bad = root/'wrong-arch'
            env.update(VLLM_SUITE_USE_EXISTING='1', MOCK_PLATFORM='linux/arm64')
            result = subprocess.run([str(package),str(bad),'suite'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('expected linux/amd64',result.stderr)
            self.assertFalse(bad.exists())
            self.assertFalse(any(line.startswith('save ') for line in (root/'calls').read_text().splitlines()))

    def test_checksum_failure_does_not_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root/'docker'
            tool.write_text('#!/bin/sh\nif [ "$1" = load ]; then touch "$LOADED_IMAGE"; fi\n')
            tool.chmod(0o755)
            archive = root/'vllm-suite-opencompass.tar.gz'
            archive.write_bytes(b'corrupt')
            (root/'SHA256SUMS').write_text('0'*64+'  ./'+archive.name+'\n')
            script = root/'load_offline.sh'
            script.write_bytes((ROOT/'scripts/load_offline.sh').read_bytes())
            script.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", LOADED_IMAGE=str(root/'loaded'))
            result = subprocess.run([str(script)],env=env,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'loaded').exists())


if __name__ == '__main__':
    unittest.main()
