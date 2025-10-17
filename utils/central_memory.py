import json, os

class CentralMemory:
    def __init__(self, path="data/memory/central_memory.json"):
        self.path = path
        self.store = {}
        if os.path.exists(path):
            with open(path) as f:
                self.store = json.load(f)

    def update(self, key, value):
        self.store[key] = value
        self.save()

    def get(self, key, default=None):
        return self.store.get(key, default)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.store, f, indent=2)

    def reset(self):
        self.store = {}
        self.save()
