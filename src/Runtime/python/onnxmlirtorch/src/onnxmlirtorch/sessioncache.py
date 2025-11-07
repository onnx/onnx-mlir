from dataclasses import dataclass
from typing import Any

@dataclass
class CacheValue:
    tag: Any = None 
    sess: Any = None 
    example_inputs_indices: Any = None

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
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    # The put is assumed to be called after victim()
    def put(self, key, value: CacheValue):
        self.cache[key] = value
        self.access_order.append(key)
        if len(self.cache) != len(self.access_order):
            print("Error: the len of cache and access_order  doesnot match")

    # Find the index of the victim entry.
    # If the cache is not full, get the next free entry
    # If the cache is full, delete the oldest key and return its index
    def victim(self):
        if len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop()
            cache_index, _ = self.cache[oldest_key]
            del self.cache[oldest_key]
            return cache_index
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
