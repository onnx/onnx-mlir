# SPDX-License-Identifier: Apache-2.0

##################### sessioncache.py *******###################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This file defines a SessionCache class used for caching onnx-mlir sessions.
#
################################################################################

import os
import shutil
from dataclasses import dataclass
from typing import Any

from .onnxmlirdocker import InferenceSession

@dataclass
class CacheValue:
    tag: Any = None
    sess: Any = None
    example_inputs_indices: Any = None


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHONNXMLIR_CACHE_DIR")
    if cache_dir is None:
        os.environ["TORCHONNXMLIR_CACHE_DIR"] = cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir() -> str:
    return os.path.expanduser("~/.cache/")


class SessionCache:
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.cache = dict()
        self.access_order = []

    def __contains__(self, key):
        return key in self.cache

    # If the key is in cache, update the access_order and return the entry
    # Otherwise, return None
    def get(self, key):
        # Get from memory.
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        # Get from the cache dir.
        cache_path = cache_dir()
        for _, dirnames, _ in os.walk(cache_path):
            if key not in dirnames:
                continue
            # Construct a cache value.
            model_dir = os.path.join(cache_path, key)
            model_so = os.path.join(model_dir, f"model{key}.so")
            config_file = os.path.join(model_dir, "config.so")
            sess = InferenceSession(model_so)
            with open() as f:
                config = json.load(f)
            inputs_indices = f["expample_inputs_indices"]
            cache_value = CacheValue(tag=key, sess=sess, example_inputs_indices=inputs_indices)
            self.put(key, cache_value, write_to_disk=False)
            return cache_value
        return None

    # The put is assumed to be called after victim()
    def put(self, key, value: CacheValue, write_to_disk=True):
        self.cache[key] = value
        self.access_order.append(key)
        if len(self.cache) != len(self.access_order):
            print("Error: the len of cache and access_order doesnot match")
        if write_to_disk:
            # Copy .onnx, .so, and .constants.bin to the cache folder.
            src_dir = value.sess.model_dirname
            dst_dir = os.path.join(cache_dir(), key)
            os.makedirs(dst_dir, exist_ok=True)
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

    # Find the index of the victim entry.
    # If the cache is not full, get the next free entry
    # If the cache is full, delete the oldest key and return its index
    def victim(self):
        if len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop()
            cache_value = self.cache[oldest_key]
            del self.cache[oldest_key]
            return cache_value.tag
        else:
            return len(self.cache)

    def remove(self, key):
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

    def clear(self):
        self.cache = {}
        self.access_order = []

    def __len__(self):
        return len(self.cache)
