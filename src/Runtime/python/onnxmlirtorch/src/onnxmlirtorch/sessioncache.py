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
import json
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
        self.cache_path = cache_dir()
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
        cache_value = self.load_from_disk(key)
        if cache_value:
            self.put(key, cache_value, write_to_disk=False)
        return cache_value

    # The put is assumed to be called after victim()
    def put(self, key, value: CacheValue, write_to_disk=True):
        self.cache[key] = value
        self.access_order.append(key)
        if len(self.cache) != len(self.access_order):
            print("Error: the len of cache and access_order doesnot match")
        if write_to_disk:
            self.write_to_disk(key, value)

    # Load data from disk into a CacheValue. If data is not found, return None.
    def load_from_disk(self, key):
        # Find a folder whose name is key and load data from that folder into a CacheValue.
        for _, dirnames, _ in os.walk(self.cache_path):
            if key not in dirnames:
                continue
            # Construct a cache value.
            model_dir = os.path.join(self.cache_path, key)
            model_so = os.path.join(model_dir, f"model{key}.so")
            config_file = os.path.join(model_dir, "config.json")
            sess = InferenceSession(model_so)
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
            except FileNotFoundError:
                config = {}
            inputs_indices = config["example_inputs_indices"] if config else []
            cache_value = CacheValue(
                tag=key, sess=sess, example_inputs_indices=inputs_indices
            )
            return cache_value
        return None

    # Write a CacheValue into a folder whose name is key.
    def write_to_disk(self, key, value: CacheValue):
        # Cache folder: create if it does not exist.
        dst_dir = os.path.join(self.cache_path, key)
        os.makedirs(dst_dir, exist_ok=True)
        # Copy the input model from the model folder to the cache folder.
        model_dir = value.sess.model_dirname
        # Copy compiled models from the output folder to the cache folder.
        output_dir = value.sess.output_dirname
        for src_dir in [model_dir, output_dir]:
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        # Create a config json file.
        config_file = os.path.join(dst_dir, "config.json")
        json_data = json.dumps(
            {"example_inputs_indices": value.example_inputs_indices},
            sort_keys=True,
            indent=4,
        )
        with open(config_file, "w") as f:
            f.write(json_data)

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
