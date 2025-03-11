class SessionCache:
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.cache = dict()
        self.access_order = []

    def __contains__(self, key):
        return key in self.cache

    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)

    def victim(self):
        if len(self.cache) >= self.capacity:
            oldest_key = self.access_order[0]
            cache_index, _ = self.cache[oldest_key]
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
