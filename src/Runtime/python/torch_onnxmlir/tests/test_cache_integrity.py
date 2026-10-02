# SPDX-License-Identifier: Apache-2.0

##################### test_cache_integrity.py ##################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Unit tests for the disk-cache integrity-check path in SessionCache.
#
# These tests exercise load_from_disk() and write_to_disk() without an actual
# onnx-mlir compiler or InferenceSession: they operate directly on
# sessioncache's helper functions and the on-disk JSON/artifact layout so the
# full adversarial scenarios run in any environment.
#
################################################################################

import hashlib
import json
import os
import stat
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch_onnxmlir
from torch_onnxmlir.sessioncache import (
    SessionCache,
    CacheValue,
    OM_BACKEND_CONFIG_FILE,
    _sha256_file,
    _makedirs_secure,
    _CACHE_DIR_MODE,
)


def _write_fake_config(model_dir: Path, so_path: Path, extra: dict = None):
    """Write an om_backend_config.json with a correct hash for *so_path*."""
    payload = {
        "artifact_hashes": {so_path.name: _sha256_file(str(so_path))},
        "example_inputs_indices": [],
        "compilation_info": "",
        "input_signature": "",
        "output_signature": "",
    }
    if extra:
        payload.update(extra)
    with open(model_dir / OM_BACKEND_CONFIG_FILE, "w") as f:
        json.dump(payload, f)


class TestCacheDirectoryPermissions(unittest.TestCase):
    """Cache directories must be created with 0o700 (owner-only)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        torch_onnxmlir.config.cache_dir = str(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        torch_onnxmlir.config.cache_dir = None

    def test_root_cache_dir_mode(self):
        from torch_onnxmlir.sessioncache import cache_dir
        path = cache_dir()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        self.assertEqual(mode, _CACHE_DIR_MODE,
                         f"cache root created with {oct(mode)}, expected {oct(_CACHE_DIR_MODE)}")

    def test_key_subdir_mode(self):
        """Subdirectory created by SessionCache.write_onnx_to_disk must be 0o700."""
        sc = SessionCache(capacity=3)
        key = "testkey_mode"
        # Create a temporary source directory with a minimal .onnx file so
        # write_onnx_to_disk has something to copy and actually creates the
        # key subdirectory via _makedirs_secure.
        with tempfile.TemporaryDirectory() as src_dir:
            (Path(src_dir) / "model.onnx").write_bytes(b"fake-onnx")
            sc.write_onnx_to_disk(key, src_dir)
        key_dir = self.tmp / key
        self.assertTrue(key_dir.exists(), "write_onnx_to_disk did not create the key subdir")
        mode = stat.S_IMODE(os.stat(key_dir).st_mode)
        self.assertEqual(mode, _CACHE_DIR_MODE,
                         f"Key subdir created with {oct(mode)}, expected {oct(_CACHE_DIR_MODE)}")

    def test_upgrade_hardens_existing_wide_open_dir(self):
        """A pre-existing 0o755 directory must be tightened to 0o700 on first use.

        os.makedirs(mode=0o700, exist_ok=True) does NOT re-apply the mode to a
        directory that already exists.  _makedirs_secure() must call os.chmod
        explicitly so that an upgrade from an older (0o755) version hardens the
        directory without the user having to delete their cache first.

        This scenario is especially relevant for non-root container users: their
        HOME is typically writable only by them anyway, but the explicit chmod
        removes any reliance on that assumption and ensures the invariant holds
        even when $HOME is on a shared volume.
        """
        wide = self.tmp / "wide_dir"
        wide.mkdir(mode=0o755)  # simulates a pre-fix version creating this dir
        self.assertEqual(stat.S_IMODE(os.stat(wide).st_mode), 0o755,
                         "Precondition: directory should start at 0o755")
        _makedirs_secure(str(wide))
        mode = stat.S_IMODE(os.stat(wide).st_mode)
        self.assertEqual(mode, _CACHE_DIR_MODE,
                         f"Existing 0o755 dir was not hardened: got {oct(mode)}")


class TestLoadFromDiskIntegrityCheck(unittest.TestCase):
    """load_from_disk must verify artifact hash before calling dlopen."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        torch_onnxmlir.config.cache_dir = str(self.tmp)
        self.sc = SessionCache(capacity=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        torch_onnxmlir.config.cache_dir = None

    def _make_cache_entry(self, key: str, so_content: bytes = b"fake-so-bytes"):
        """Plant a cache directory with a model.so and matching config."""
        model_dir = self.tmp / key
        model_dir.mkdir(mode=_CACHE_DIR_MODE, parents=True, exist_ok=True)
        so_path = model_dir / "model.so"
        so_path.write_bytes(so_content)
        _write_fake_config(model_dir, so_path)
        return model_dir, so_path

    def test_valid_entry_loads(self):
        """A correctly-hashed entry triggers InferenceSession construction."""
        key = "valid_key"
        self._make_cache_entry(key)
        fake_sess = MagicMock()
        with patch(
            "torch_onnxmlir.sessioncache.InferenceSession", return_value=fake_sess
        ):
            result = self.sc.load_from_disk(key)
        self.assertIsNotNone(result, "Expected a CacheValue for a valid entry")
        self.assertIs(result.sess, fake_sess)

    def test_tampered_so_returns_none(self):
        """A poisoned .so (wrong bytes) must be refused — cache miss, not load."""
        key = "tampered_key"
        model_dir, so_path = self._make_cache_entry(key)
        # Overwrite the .so with different bytes after the hash was written.
        so_path.write_bytes(b"TAMPERED-PAYLOAD")
        with patch("torch_onnxmlir.sessioncache.InferenceSession") as mock_sess:
            result = self.sc.load_from_disk(key)
        self.assertIsNone(result,
                          "Tampered .so must produce a cache miss (None), not a loaded session")
        mock_sess.assert_not_called()

    def test_no_hash_in_config_returns_none(self):
        """Pre-fix config with no artifact_hashes must be treated as cache miss."""
        key = "legacy_key"
        model_dir = self.tmp / key
        model_dir.mkdir(mode=_CACHE_DIR_MODE, parents=True, exist_ok=True)
        (model_dir / "model.so").write_bytes(b"some-so-bytes")
        # Write a config without 'artifact_hashes' (simulates pre-fix version).
        legacy_config = {
            "example_inputs_indices": [],
            "compilation_info": "",
            "input_signature": "",
            "output_signature": "",
        }
        with open(model_dir / OM_BACKEND_CONFIG_FILE, "w") as f:
            json.dump(legacy_config, f)
        with patch("torch_onnxmlir.sessioncache.InferenceSession") as mock_sess:
            result = self.sc.load_from_disk(key)
        self.assertIsNone(result,
                          "Legacy config without artifact_hashes must be a cache miss")
        mock_sess.assert_not_called()

    def test_missing_config_returns_none(self):
        """No config file at all must be treated as cache miss."""
        key = "no_config_key"
        model_dir = self.tmp / key
        model_dir.mkdir(mode=_CACHE_DIR_MODE, parents=True, exist_ok=True)
        (model_dir / "model.so").write_bytes(b"some-so-bytes")
        with patch("torch_onnxmlir.sessioncache.InferenceSession") as mock_sess:
            result = self.sc.load_from_disk(key)
        self.assertIsNone(result,
                          "Missing config must be a cache miss")
        mock_sess.assert_not_called()

    def test_missing_model_so_returns_none(self):
        """A key directory that exists but has no model.so must be a cache miss."""
        key = "no_so_key"
        model_dir = self.tmp / key
        model_dir.mkdir(mode=_CACHE_DIR_MODE, parents=True, exist_ok=True)
        with patch("torch_onnxmlir.sessioncache.InferenceSession") as mock_sess:
            result = self.sc.load_from_disk(key)
        self.assertIsNone(result)
        mock_sess.assert_not_called()

    def test_path_traversal_key_returns_none(self):
        """A config with a path-traversal key must be a cache miss, not a file read."""
        key = "traversal_key"
        model_dir = self.tmp / key
        model_dir.mkdir(mode=_CACHE_DIR_MODE, parents=True, exist_ok=True)
        (model_dir / "model.so").write_bytes(b"some-so-bytes")
        # Plant a config whose artifact_hashes key escapes the cache directory.
        bad_config = {
            "artifact_hashes": {"../outside/model.so": "not-a-real-hash"},
            "example_inputs_indices": [],
            "compilation_info": "",
            "input_signature": "",
            "output_signature": "",
        }
        with open(model_dir / OM_BACKEND_CONFIG_FILE, "w") as f:
            json.dump(bad_config, f)
        with patch("torch_onnxmlir.sessioncache.InferenceSession") as mock_sess:
            result = self.sc.load_from_disk(key)
        self.assertIsNone(result, "Path-traversal key must be a cache miss")
        mock_sess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
